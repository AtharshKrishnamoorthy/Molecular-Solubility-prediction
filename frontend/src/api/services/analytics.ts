import type { AnalyticsRecord, CreateAnalyticsBody } from "../types";

const API_URL = process.env.NEXT_PUBLIC_DASHBOARD_API_URL ?? "http://localhost:8002";

async function request<T>(method: string, path: string, body?: object): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

export function createAnalytics(body: CreateAnalyticsBody): Promise<AnalyticsRecord> {
  return request("POST", "/analytics", body);
}

export function getAnalyticsByUser(userId: string): Promise<AnalyticsRecord[]> {
  return request("GET", `/analytics/user/${userId}`);
}

export function getAnalyticsById(analyticsId: string): Promise<AnalyticsRecord> {
  return request("GET", `/analytics/${analyticsId}`);
}

export function deleteAnalytics(analyticsId: string): Promise<{ deleted: boolean }> {
  return request("DELETE", `/analytics/${analyticsId}`);
}

export function deleteAllUserAnalytics(userId: string): Promise<{ deleted_count: number }> {
  return request("DELETE", `/analytics/user/${userId}/all`);
}
