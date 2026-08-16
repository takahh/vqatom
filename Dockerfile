FROM node:22-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 7860
CMD ["npm", "run", "start", "--", "--host", "0.0.0.0", "--port", "7860"]
