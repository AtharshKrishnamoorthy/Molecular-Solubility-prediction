import type { User, CreateUserBody, UpdateEmailBody, UpdatePasswordBody } from "../types";

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

export function getUserById(userId: string): Promise<User> {
  return request("GET", `/users/${userId}`);
}

export function getUserByEmail(email: string): Promise<User> {
  return request("GET", `/users/by-email/${encodeURIComponent(email)}`);
}

export function createUser(body: CreateUserBody): Promise<User> {
  return request("POST", "/users", body);
}

export function updateUserEmail(userId: string, body: UpdateEmailBody): Promise<User> {
  return request("PATCH", `/users/${userId}/email`, body);
}

export function updateUserPassword(userId: string, body: UpdatePasswordBody): Promise<User> {
  return request("PATCH", `/users/${userId}/password`, body);
}

export function deleteUser(userId: string): Promise<{ deleted: boolean }> {
  return request("DELETE", `/users/${userId}`);
}
