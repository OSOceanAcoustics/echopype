FROM minio/minio

# Install git
RUN microdnf install git

CMD ["minio", "server", "/data"]