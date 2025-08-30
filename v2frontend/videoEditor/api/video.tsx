// api/video.ts
import { API_URL } from "./config";

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const ac = new AbortController();
  const id = setTimeout(() => ac.abort(), 15000);

  try {
    const headers =
      init.body instanceof FormData
        ? init.headers                                 // let browser set multipart boundary
        : { "Content-Type": "application/json", ...(init.headers || {}) };
    // console.log("API Request:", path, init.method || "GET", init.body);
    console.log("Full URL:", `${API_URL}${path}`);
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: ac.signal, headers });
    console.log("Full URL:", `${res}`);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
       throw new Error(`HTTP ${res.status} ${text}`);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  } finally {
    clearTimeout(id);
  }
}

export const api = {
  // JSON submit (if you still need it elsewhere)
  submitVideoJson: (body: any) =>
    request<{ jobId: string }>("/jobs", { method: "POST", body: JSON.stringify(body) }),

  // Proper file upload
  uploadVideo: (form: FormData) =>
    request<{ jobId: string, videoUri: string, pill_objects: any[] }>("/detect-pii", { method: "POST", body: form }),

  getJob: (jobId: string) =>
    request<{ status: string; result?: any }>(`/jobs/${jobId}`, { method: "GET" }),
};
