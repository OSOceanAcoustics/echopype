FROM minio/minio
ARG TARGETPLATFORM

# Install git
RUN microdnf install git

CMD ["minio", "server", "/data"]
