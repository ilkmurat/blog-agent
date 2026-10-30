#!/bin/sh
# Generates langfuse/.env with random secrets and points ../.env (blog-agent) at this Langfuse instance.
set -eu
cd "$(dirname "$0")"
[ -f .env ] && { echo "langfuse/.env already exists; not overwriting"; exit 0; }
hex() { openssl rand -hex "$1"; }
PG=$(hex 16); CH=$(hex 16); RD=$(hex 16); MN=$(hex 16)
PK="pk-lf-$(hex 16)"; SK="sk-lf-$(hex 16)"; PW=$(hex 8)
cat > .env <<ENV
COMPOSE_PROJECT_NAME=langfuse
NEXTAUTH_URL=http://localhost:3001
TELEMETRY_ENABLED=false
NEXTAUTH_SECRET=$(hex 32)
SALT=$(hex 32)
ENCRYPTION_KEY=$(hex 32)
POSTGRES_PASSWORD=$PG
DATABASE_URL=postgresql://postgres:$PG@postgres:5432/postgres
CLICKHOUSE_PASSWORD=$CH
REDIS_AUTH=$RD
MINIO_ROOT_PASSWORD=$MN
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=$MN
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=$MN
LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY=$MN
LANGFUSE_INIT_ORG_ID=inkwell
LANGFUSE_INIT_ORG_NAME=Inkwell
LANGFUSE_INIT_PROJECT_ID=blog-agent
LANGFUSE_INIT_PROJECT_NAME=blog-agent
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=$PK
LANGFUSE_INIT_PROJECT_SECRET_KEY=$SK
LANGFUSE_INIT_USER_EMAIL=admin@inkwell.local
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=$PW
ENV
chmod 600 .env
# blog-agent reads its keys from ../.env (the agent container reaches Langfuse on the host)
touch ../.env
grep -v '^LANGFUSE_' ../.env > ../.env.tmp || true
cat >> ../.env.tmp <<ENV
LANGFUSE_PUBLIC_KEY=$PK
LANGFUSE_SECRET_KEY=$SK
LANGFUSE_BASE_URL=http://host.docker.internal:3001
ENV
mv ../.env.tmp ../.env && chmod 600 ../.env
echo "Langfuse UI: http://localhost:3001  login: admin@inkwell.local / $PW"
