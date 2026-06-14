# Embedded Mastery — tiny static site served by nginx.
# Everything is offline-bundled, so this image needs no network at runtime.
FROM nginx:1.27-alpine

# our server config (gzip, long-cache for assets, correct font MIME types)
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

# the course itself (index.html, Arduino/, STM32/, assets/)
COPY index.html        /usr/share/nginx/html/index.html
COPY Arduino/          /usr/share/nginx/html/Arduino/
COPY STM32/            /usr/share/nginx/html/STM32/
COPY assets/           /usr/share/nginx/html/assets/

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
