const CACHE_NAME = 'almaty-air-v1';
const ASSETS = [
  '/',
  '/static/css/style.css', // укажи свои пути к CSS/JS
  '/static/manifest.json'
];

// Установка: кэшируем базовые файлы
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

// Активация: чистим старый кэш
self.addEventListener('activate', (event) => {
  console.log('Service Worker activated');
});

// ОБРАБОТКА PUSH-УВЕДОМЛЕНИЙ
self.addEventListener('push', (event) => {
  let data = { title: 'Воздух Алматы', body: 'Обновление данных' };
  
  if (event.data) {
    data = event.data.json();
  }

  const options = {
    body: data.body,
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/icon-192x192.png',
    vibrate: [100, 50, 100],
    data: {
      url: data.url || '/'
    }
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

// Клик по уведомлению открывает приложение
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow(event.notification.data.url)
  );
});