FROM minio/minio
ARG TARGETPLATFORM

# Install git
USER root
RUN microdnf install git

CMD ["minio", "server", "/data"]
