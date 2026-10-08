import axios from "axios";

// Central axios instance. Vite dev server proxies /api to the BFF (see vite.config.js).
const api = axios.create({ baseURL: "/api" });

export default api;
