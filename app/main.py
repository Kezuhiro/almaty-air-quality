import os
import math
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# 1. Импортируем только КЛАССЫ моделей (без глобальных объектов)
from services.catboost_service import CatBoostRunner
from services.stgcn_service import STGCNRunner

# 2. Импортируем сервисы и менеджеры
from services.fetcher import om_fetcher, openaq_fetcher
from services.history_service import history_service
from services.data_manager import data_manager

class SimulationRequest(BaseModel):
    temp: float = Field(..., description="Температура в Цельсиях")
    wind: float = Field(..., description="Скорость ветра м/с")
    direction: float = Field(..., description="Метеорологическое направление ветра (0-360)")
    humidity: float = Field(..., description="Влажность %")
    hour: int = Field(..., ge=0, le=23, description="Час суток (0-23)")

# Глобальное хранилище для моделей в оперативной памяти
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ==========================================
    # ВЫПОЛНЯЕТСЯ ОДИН РАЗ ПРИ СТАРТЕ СЕРВЕРА
    # ==========================================
    print("Инициализация ML-движков в оперативную память...")
    
    # 1. Загружаем модели строго 1 раз в словарь
    ml_models["catboost"] = CatBoostRunner()
    ml_models["stgcn"] = STGCNRunner()
    
    # 2. Делаем первый прогревочный запрос данных
    print("Прогрев кэша API...")
    openaq_fetcher.get_realtime_24_nodes()
    await om_fetcher.get_current_weather()
    
    print("🚀 Сервер готов к работе! Все модели в RAM.")
    
    yield # Здесь сервер работает и принимает запросы
    
    # ==========================================
    # ВЫПОЛНЯЕТСЯ ПРИ ВЫКЛЮЧЕНИИ СЕРВЕРА
    # ==========================================
    print("Очистка оперативной памяти...")
    ml_models.clear()


app = FastAPI(
    title="Almaty Air Quality API",
    version="1.0",
    description="API для мониторинга и прогнозирования качества воздуха в Алматы",
    lifespan=lifespan
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


async def _build_home_data():
    last_features = data_manager.get_last_features()
    city_data = await om_fetcher.get_city_average_history()
    
    try:
        # ОБРАЩАЕМСЯ СТРОГО ЧЕРЕЗ ml_models
        prediction = await ml_models["catboost"].predict_tomorrow(
            current_pm25=city_data.get("current_pm25", 0),
            current_pm10=city_data.get("current_pm10", 0),
            roll7_pm25=city_data.get("roll7_pm25", 0)
        )
    except Exception as e:
        print(f"Ошибка CatBoost на главной: {e}")
        prediction = None

    current_weather = await om_fetcher.get_current_weather()

    return {
        "current_pm25": round(city_data.get("current_pm25", 0), 1),
        "tomorrow_pm25": round(prediction["pm25"], 1) if prediction else round(city_data.get("roll7_pm25", 0), 1),
        "temp": current_weather["temp"],
        "wind": current_weather["wind"],
        "updated_at": city_data.get("updated_at", "сейчас"),
        "trend_labels": city_data.get("trend_labels", []),
        "trend_values": city_data.get("trend_values", [])
    }


def _build_stgcn_stations(city_data):
    stations = []
    try:
        # ОБРАЩАЕМСЯ СТРОГО ЧЕРЕЗ ml_models
        predicted_values = ml_models["stgcn"].get_predictions()
        stgcn_stations_info = ml_models["stgcn"].stations
        
        current_nodes = openaq_fetcher.cached_24_nodes
        if not current_nodes or len(current_nodes) != len(stgcn_stations_info):
            current_nodes = [city_data.get("current_pm25", 0)] * len(stgcn_stations_info)
            
        sources = openaq_fetcher.cached_24_sources or ["imputed"] * len(stgcn_stations_info)
        
        for index, station in enumerate(stgcn_stations_info):
            curr_val = float(current_nodes[index])
            pred_val = float(predicted_values[index])
            
            stations.append({
                "name": station["name"],
                "lat": station["lat"],
                "lon": station["lon"],
                "current_pm25": round(curr_val, 1),
                "pm25": round(pred_val, 1), 
                "delta": round(pred_val - curr_val, 1), 
                "source": sources[index] if index < len(sources) else "imputed"
            })
    except Exception as exc:
        print(f"STGCN station payload error: {exc}")
        # Если произошла ошибка, используем запасной план (fallback)
        fallback_stations = ml_models["stgcn"].stations if "stgcn" in ml_models else []
        for station in fallback_stations:
            stations.append({
                "name": station["name"],
                "lat": station["lat"],
                "lon": station["lon"],
                "current_pm25": round(city_data.get("current_pm25", 0), 1),
                "pm25": round(city_data.get("current_pm25", 0), 1),
                "delta": 0.0,
                "source": "imputed"
            })
    return stations


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _pm25_risk_band(value: float):
    if value <= 15:
        return ("Низкий риск", "success")
    if value <= 35:
        return ("Умеренный фон", "info")
    if value <= 55:
        return ("Неблагоприятно", "warning")
    return ("Высокий риск", "danger")


def _confidence_band(score: int):
    if score >= 80:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "title": "Главная",
            "active_page": "home",
            "home_data": await _build_home_data()
        }
    )


@app.get("/forecast", response_class=HTMLResponse)
async def page_forecast(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(data_manager.update_dataset)

    try:
        last_features = data_manager.get_last_features()
        lag25 = last_features.get('pm25', 0)
        lag10 = last_features.get('pm10', 0)
        roll7 = last_features.get('pm25_roll7_mean', 0)

        # ОБРАЩАЕМСЯ СТРОГО ЧЕРЕЗ ml_models
        prediction_data = await ml_models["catboost"].predict_tomorrow(
            current_pm25=lag25, 
            current_pm10=lag10, 
            roll7_pm25=roll7
        )

        city_data = await om_fetcher.get_city_average_history()
        current_weather = await om_fetcher.get_current_weather()

        prediction_data["current_pm25"] = round(city_data.get("current_pm25", 0), 1)
        prediction_data["current_pm10"] = round(city_data.get("current_pm10", 0), 1)
        prediction_data["temp"] = current_weather["temp"]
        prediction_data["wind"] = current_weather["wind"]

        stgcn_stations_list = _build_stgcn_stations(city_data)

        return templates.TemplateResponse(
            request=request,
            name="forecast.html", 
            context={
                "title": "Прогноз",
                "active_page": "forecast",
                "data": prediction_data,
                "stgcn_stations": stgcn_stations_list 
            }
        )

    except Exception as e:
        print(f"Ошибка на роуте /forecast: {e}")
        return templates.TemplateResponse(
            request=request,
            name="forecast.html", 
            context={
                "title": "Прогноз",
                "active_page": "forecast",
                "error": "Не удалось сформировать прогноз. Проверьте подключение к API или данные в CSV."
            }
        )


@app.get("/map", response_class=HTMLResponse)
async def get_map():
    return HTMLResponse(content=ml_models["stgcn"].generate_heatmap())


@app.post("/api/simulate")
async def api_simulate(req: SimulationRequest):
    try:
        base_pm25 = openaq_fetcher.get_realtime_24_nodes()
        stations = ml_models["stgcn"].stations
        
        sim_preds = ml_models["stgcn"].simulate_scenario(
            temp=req.temp, wind=req.wind, direction=req.direction, hum=req.humidity, hour=req.hour
        )
        
        avg_base = sum(base_pm25) / len(base_pm25) if base_pm25 else 1.0
        avg_sim = sum(sim_preds) / len(sim_preds) if sim_preds else 1.0
        global_delta = avg_sim - avg_base
        anomaly_score = abs(req.temp - 15) + abs(req.wind - 2) * 5 + max(0, req.humidity - 70) * 0.2
        confidence_score = int(round(_clamp(100 - anomaly_score * 2.4, 18, 97)))
        confidence_label = _confidence_band(confidence_score)

        results = []
        for i in range(len(stations)):
            b_val = float(base_pm25[i]) if base_pm25 else 0.0
            s_val = float(sim_preds[i]) if sim_preds else 0.0

            results.append({
                "id": i, "name": stations[i]["name"], 
                "lat": stations[i]["lat"], "lon": stations[i]["lon"],
                "base_pm25": round(b_val, 1), "sim_pm25": round(s_val, 1), 
                "delta": round(s_val - b_val, 1), "confidence": confidence_label
            })

        edges = []
        rad = math.radians(req.direction)
        wx = -math.sin(rad)
        wy = -math.cos(rad) 
        
        for i in range(len(stations)):
            for j in range(len(stations)):
                if i != j:
                    dx = stations[j]["lon"] - stations[i]["lon"]
                    dy = stations[j]["lat"] - stations[i]["lat"]
                    dist = math.sqrt(dx**2 + dy**2)
                    
                    if 0.01 < dist < 0.08: 
                        ndx, ndy = dx/dist, dy/dist
                        dot = wx * ndx + wy * ndy 
                        
                        if dot > 0.7 and req.wind > 0.5: 
                            edges.append({
                                "source": i, "target": j, 
                                "weight": round(dot * req.wind, 2)
                            })

        insights = []
        remaining_delta = global_delta
        
        if req.wind < 1.0:
            w_impact = max(1.0, remaining_delta * 0.5) if remaining_delta > 0 else 5.0
            insights.append({"factor": "Штиль / Стагнация", "impact": "danger", "weight": round(w_impact, 1), "desc": "Отсутствие ветра блокирует вынос аэрозолей из города."})
        elif req.wind > 4.0:
            w_impact = min(-1.0, remaining_delta * 0.6) if remaining_delta < 0 else -8.0
            insights.append({"factor": "Ветровая вентиляция", "impact": "success", "weight": round(w_impact, 1), "desc": "Смог активно выдувается за пределы городской застройки."})
            
        if req.temp <= 0:
            t_impact = max(1.0, remaining_delta * 0.4) if remaining_delta > 0 else 8.0
            insights.append({"factor": "Выбросы ТЭЦ и печей", "impact": "danger", "weight": round(t_impact, 1), "desc": "Пиковая нагрузка на теплосети из-за низких температур."})
        elif req.temp > 25:
            t_impact = min(-1.0, remaining_delta * 0.2) if remaining_delta < 0 else -3.0
            insights.append({"factor": "Тепловая конвекция", "impact": "success", "weight": round(t_impact, 1), "desc": "Нагрев поверхности помогает частицам подниматься вверх."})
            
        if req.hour in [8, 9, 18, 19]:
            h_impact = max(1.0, remaining_delta * 0.3) if remaining_delta > 0 else 6.0
            insights.append({"factor": "Час пик (Трафик)", "impact": "warning", "weight": round(h_impact, 1), "desc": "Резкий рост выхлопных газов от автомобилей."})
            
        if req.humidity > 75 and req.wind < 2.0 and req.temp < 10:
            i_impact = max(1.0, remaining_delta * 0.2) if remaining_delta > 0 else 4.0
            insights.append({"factor": "Температурная инверсия", "impact": "danger", "weight": round(i_impact, 1), "desc": "Холодный влажный воздух прижат к земле, создавая 'купол'."})

        if not insights and abs(global_delta) > 1.0:
            insights.append({"factor": "Фоновое рассеивание", "impact": "success" if global_delta < 0 else "warning", "weight": round(global_delta, 1), "desc": "Естественная динамика атмосферы без критических факторов."})

        insights.sort(key=lambda x: abs(x["weight"]), reverse=True)

        sorted_by_pm25 = sorted(results, key=lambda item: item["sim_pm25"], reverse=True)
        sorted_by_delta_up = sorted(results, key=lambda item: item["delta"], reverse=True)
        sorted_by_delta_down = sorted(results, key=lambda item: item["delta"])

        worsened_count = sum(1 for item in results if item["delta"] > 1.0)
        improved_count = sum(1 for item in results if item["delta"] < -1.0)
        stable_count = len(results) - worsened_count - improved_count

        risk_label, risk_tone = _pm25_risk_band(avg_sim)
        scenario_tags = []
        if req.wind < 1.0:
            scenario_tags.append("Штиль")
        elif req.wind > 5.0:
            scenario_tags.append("Сильная вентиляция")
        if req.hour in [8, 9, 18, 19]:
            scenario_tags.append("Час пик")
        if req.temp <= 0:
            scenario_tags.append("Отопительный режим")
        if req.humidity >= 75:
            scenario_tags.append("Высокая влажность")
        if not scenario_tags:
            scenario_tags.append("Фоновое рассеивание")

        summary = {
            "avg_base_pm25": round(avg_base, 1),
            "avg_sim_pm25": round(avg_sim, 1),
            "global_delta": round(global_delta, 1),
            "risk_label": risk_label,
            "risk_tone": risk_tone,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "worsened_count": worsened_count,
            "improved_count": improved_count,
            "stable_count": stable_count,
            "wind_links": len(edges),
            "scenario_tags": scenario_tags,
            "hotspot": sorted_by_pm25[0] if sorted_by_pm25 else None,
            "cleanest": sorted(results, key=lambda item: item["sim_pm25"])[0] if results else None,
            "largest_worsening": sorted_by_delta_up[0] if sorted_by_delta_up else None,
            "largest_improvement": sorted_by_delta_down[0] if sorted_by_delta_down else None
        }

        return {
            "status": "success", 
            "data": results, 
            "insights": insights,
            "edges": edges, 
            "global_delta": round(global_delta, 1),
            "summary": summary,
            "top_hotspots": sorted_by_pm25[:5],
            "top_worsening": sorted_by_delta_up[:5],
            "top_improvements": sorted_by_delta_down[:5]
        }
    except Exception as e:
        import traceback
        print(f"Ошибка Backend API:\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@app.get("/simulation", response_class=HTMLResponse)
async def page_simulation(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="simulation.html", 
        context={
            "title": "3D Симуляция",
            "active_page": "simulation"
        }
    )


@app.get("/history", response_class=HTMLResponse)
async def page_history(request: Request):
    context = history_service.get_context()
    context["title"] = "История"
    context["active_page"] = "history"
    
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context=context
    )


@app.get("/about", response_class=HTMLResponse)
async def page_about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"title": "О проекте", "active_page": "about"}
    )


@app.get("/api/predict/{model_name}")
async def get_prediction(model_name: str):
    if model_name == "stgcn":
        return {"status": "success", "model": "STGCN", "message": "Здесь будет прогноз GNN на 3 часа"}
    if model_name == "catboost":
        return {"status": "success", "model": "CatBoost", "message": "Здесь будет суточный прогноз"}
    return {"status": "error", "message": "Модель не найдена"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)