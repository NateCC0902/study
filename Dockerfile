# Study — unified offline learning site (Embedded · ML/DL · PID), served by nginx.
# Everything is offline-bundled, so this image needs no network at runtime.
FROM nginx:1.27-alpine

# our server config (gzip, long-cache for assets, correct font MIME types)
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf

# the whole site: master hub + shared assets + the three courses
COPY index.html         /usr/share/nginx/html/index.html
COPY assets/            /usr/share/nginx/html/assets/
COPY embedded-mastery/  /usr/share/nginx/html/embedded-mastery/
COPY ml-dl-curriculum/  /usr/share/nginx/html/ml-dl-curriculum/
COPY pid/               /usr/share/nginx/html/pid/

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
