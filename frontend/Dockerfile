# Build Stage
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production Stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Copy custom nginx config if needed, otherwise default is okay for SPA 
# (might need try_files for React Router if we used it, but we are single page for now)
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
