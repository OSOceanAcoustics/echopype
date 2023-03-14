FROM httpd:2.4
ARG TARGETPLATFORM

COPY echopype/test_data /usr/local/apache2/htdocs/data
