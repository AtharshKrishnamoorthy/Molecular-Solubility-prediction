import type { SignupResponse, SigninResponse } from "../types";

const AUTH_URL = process.env.NEXT_PUBLIC_AUTH_API_URL ?? "http://localhost:8001";

async function post<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${AUTH_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

export function signup(email: string, password: string): Promise<SignupResponse> {
  return post("/signup", { email, password });
}

export function signin(email: string, password: string): Promise<SigninResponse> {
  return post("/signin", { email, password });
}
