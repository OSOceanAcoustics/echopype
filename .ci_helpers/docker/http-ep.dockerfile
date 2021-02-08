FROM httpd:2.4

RUN echo "Installing Apt-get packages..." \
    && apt-get update --fix-missing \
    && apt-get install -y apt-utils 2> /dev/null \
    && apt-get install -y wget zip tzdata \
    && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./public-html/ /usr/local/apache2/htdocs/
COPY ./download.sh /srv/download.sh

ENTRYPOINT ["/srv/download.sh"]
CMD ["httpd-foreground"]