# ---- Base Stage ----
FROM node:23-alpine AS base
WORKDIR /app

RUN npm i -g typescript

# ---- Development Stage ----
FROM base AS development

# For development, we just copy everything and do a regular npm install
COPY ./package.json ./
COPY package-lock.json ./

RUN npm install

COPY ./ ./

#ENV PATH /app/node_modules/.bin:$PATH

EXPOSE 3001

CMD ["npm", "run", "dev:host"]

# ---- Production Dependencies Stage ----
FROM base AS prod-deps
WORKDIR /app

RUN npm i -g npm@latest

COPY ./package.json ./
COPY package-lock.json ./

#RUN npm install

COPY ./package* ./

RUN npm ci --omit=dev
#RUN npm run postinstall

# ---- Build Stage ----
FROM prod-deps AS build
COPY ./ ./

ARG NODE_ENV=production
ENV NODE_ENV=${NODE_ENV}

# Redeclare arguments for runtime
ARG NODE_ENV
ARG VITE_APP_KEYCLOAK_REALM
ARG VITE_APP_KEYCLOAK_CLIENTID
ARG VITE_APP_KEYCLOAK_URL
ARG VITE_APP_API_BASE_URL

# Set environment variables for runtime
ENV NODE_ENV=$NODE_ENV
ENV VITE_APP_KEYCLOAK_REALM=$VITE_APP_KEYCLOAK_REALM
ENV VITE_APP_KEYCLOAK_CLIENTID=$VITE_APP_KEYCLOAK_CLIENTID
ENV VITE_APP_KEYCLOAK_URL=$VITE_APP_KEYCLOAK_URL
ENV VITE_APP_API_BASE_URL=$VITE_APP_API_BASE_URL

ENV PATH /app/node_modules/.bin:$PATH

COPY ./load_env_build.sh ./
CMD ["sh", "load_env_build.sh"]
