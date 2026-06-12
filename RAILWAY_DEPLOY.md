# Deploy en Railway

## Start command

Railway usara el `Procfile`:

```Procfile
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

No configures `PORT` manualmente; Railway lo inyecta.

## Variables

Configura estas variables en el servicio:

```text
FOOTBALL_DATA_API_KEY=tu_api_key
```

`PREDICTIONS_FILE` es opcional. Si no lo defines, la app usara `/app/storage/predictions.json` cuando exista ese Volume, o `predictions.json` localmente.

## Volume

Para conservar predicciones despues de redeploy/restart:

1. Crea un Railway Volume conectado al servicio.
2. Montalo en `/app/storage`.
3. Redeploy.
4. Guarda una prediccion y reinicia/redeploy para confirmar que sigue presente.

## Archivos ignorados

`.railwayignore` evita subir secretos, cache local y datasets crudos pesados que no son necesarios para arrancar la app.

## PWA

La app incluye `static/manifest.json`, iconos y `static/sw.js`. Para que el telefono la trate como instalable, usa el dominio HTTPS publico de Railway o un dominio propio con HTTPS.

Streamlit necesita conexion activa con el servidor para ejecutar la app; el service worker solo cachea assets basicos de instalacion, no convierte el predictor en modo offline completo.
