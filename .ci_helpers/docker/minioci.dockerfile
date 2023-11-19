FROM minio/minio
ARG TARGETPLATFORM

CMD ["minio", "server", "/data"]
