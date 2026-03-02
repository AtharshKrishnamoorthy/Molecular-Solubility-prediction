"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, Eye, EyeOff } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { DotBackground } from "@/components/dot-background";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { signin } from "@/api/services/auth";

export default function SigninPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email || !password) return;
    setLoading(true);
    try {
      const result = await signin(email, password);
      localStorage.setItem("access_token", result.access_token);
      localStorage.setItem("user_id", result.user_id);
      localStorage.setItem("user_email", email);
      toast.success("Signed in successfully!");
      router.push("/dashboard");
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Sign in failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <DotBackground>
      <div className="min-h-screen flex flex-col">
        {/* Nav */}
        <nav className="border-b border-border px-4 sm:px-6 py-4 flex items-center justify-between backdrop-blur-sm bg-background/80 sticky top-0 z-50">
          <Link href="/" className="flex items-center gap-2.5">
            <Logo width={24} height={24} />
            <span className="font-semibold tracking-tight text-sm">MolSol</span>
          </Link>
          <ThemeToggle />
        </nav>

        {/* Form */}
        <div className="flex flex-1 items-center justify-center px-4 py-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="w-full max-w-sm"
          >
            <Card className="border-border bg-card/80 backdrop-blur-sm shadow-lg">
              <CardHeader className="pb-5 pt-6 px-6">
                <div className="flex items-center gap-2.5 mb-3">
                  <Logo width={28} height={28} />
                </div>
                <CardTitle className="text-xl tracking-tight">Welcome back</CardTitle>
                <CardDescription className="text-sm">Sign in to your MolSol account.</CardDescription>
              </CardHeader>
              <CardContent className="px-6 pb-6">
                <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1.5">
                    <Label htmlFor="email" className="text-xs font-medium">Email</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder="you@example.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      autoFocus
                      className="h-10"
                    />
                  </div>

                  <div className="flex flex-col gap-1.5">
                    <Label htmlFor="password" className="text-xs font-medium">Password</Label>
                    <div className="relative">
                      <Input
                        id="password"
                        type={showPassword ? "text" : "password"}
                        placeholder="Your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        className="h-10 pr-10"
                      />
                      <button
                        type="button"
                        onClick={() => setShowPassword((v) => !v)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                        aria-label={showPassword ? "Hide password" : "Show password"}
                        tabIndex={-1}
                      >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>

                  <Button type="submit" disabled={loading} className="w-full gap-2 h-10 mt-1 font-medium">
                    {loading && <Loader2 className="w-4 h-4 animate-spin" />}
                    {loading ? "Signing in..." : "Sign In"}
                  </Button>
                </form>

                <p className="text-xs text-muted-foreground text-center mt-5">
                  Don&apos;t have an account?{" "}
                  <Link href="/auth/signup" className="text-foreground font-medium underline underline-offset-4 hover:text-primary transition-colors">
                    Sign up
                  </Link>
                </p>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </DotBackground>
  );
}
