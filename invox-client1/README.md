# AI Concierge Client QA

## Prerequisites

- Node.js LTS
- Docker

## Used packages

- [React](https://react.dev/)
- [Vite](https://vitejs.dev/)
- [Biome](https://biomejs.dev/)
- [Tanstack Query](https://tanstack.com/query)
- [axios](https://axios-http.com/)
- [React Router](https://reactrouter.com/)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn/ui](https://ui.shadcn.com/)
- [i18next](https://www.i18next.com/)
- [react-oidc-context](https://github.com/authts/react-oidc-context?tab=readme-ov-file#documentation)

## How to run

### Install dependencies

```bash
npm install
```

### Docker

We use docker compose to run the application both in development and production mode.
Development runs on port 3001 with hot reloading so you can edit the code and see the changes without rebuilding the image.

#### Development

From the tapir-services directory:

```bash
docker compose -f docker-compose.dev.yml up --build
```

#### Build

From the dd-services directory:

```bash
docker compose up
```

#### Format and lint

From the dd-client directory:

```bash
npm run check
```

## Environment variables

Be sure to include a `.env` file in the root of the dd-services directory with the environment variables from `.env.example`.

### List of variables

| Name | Description |
| ---- | ----------- |
| VITE_APP_KEYCLOAK_REALM | The realm of the Keycloak instance. |
| VITE_APP_KEYCLOAK_CLIENTID | The client ID of the application in Keycloak. |
| VITE_APP_API_BASE_URL | The base URL of the API. |
