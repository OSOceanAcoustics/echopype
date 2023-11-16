FROM minio/minio:RELEASE.2023-10-25T06-33-25Z.fips
ARG TARGETPLATFORM

# Install git
RUN microdnf install git

CMD ["minio", "server", "/data"]
