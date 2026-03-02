"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowLeft,
  Loader2,
  Atom,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Beaker,
  PanelLeftClose,
  PanelLeftOpen,
  Menu,
  FlaskConical,
  History,
  LogOut,
  User,
  Clock,
  Trash2,
  RotateCcw,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { analyzeSmiles } from "@/api/api";
import { createAnalytics, getAnalyticsByUser, deleteAnalytics } from "@/api/services/analytics";
import type { MoleculeResult, AnalyticsRecord } from "@/api/types";

// ---- helpers ----

function logSColor(val: number | null): string {
  if (val === null) return "text-muted-foreground";
  if (val >= -1) return "text-green-400";
  if (val >= -3) return "text-yellow-400";
  return "text-red-400";
}

function logSLabel(val: number | null): string {
  if (val === null) return "---";
  if (val >= -1) return "Highly Soluble";
  if (val >= -3) return "Moderately Soluble";
  return "Poorly Soluble";
}

const DEFAULT_SMILES = "NCCCC\nCCC\nCN";

// ---- Sidebar content (shared between desktop + mobile) ----

function SidebarBody({
  input,
  setInput,
  loading,
  onAnalyze,
}: {
  input: string;
  setInput: (v: string) => void;
  loading: boolean;
  onAnalyze: () => void;
}) {
  return (
    <div className="flex flex-col gap-5 p-5">
      <div>
        <Button asChild variant="ghost" size="sm" className="-ml-2 mb-4 text-muted-foreground hover:text-foreground">
          <Link href="/">
            <ArrowLeft className="w-3.5 h-3.5 mr-1" />
            Home
          </Link>
        </Button>

        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mb-2">
          Input SMILES
        </p>
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={"NCCCC\nCCC\nCN"}
          className="font-mono text-xs resize-none h-36 bg-background/60 border-border focus-visible:ring-1"
          spellCheck={false}
        />
        <p className="text-[11px] text-muted-foreground mt-1.5">One SMILES per line.</p>
      </div>

      <Button onClick={onAnalyze} disabled={loading} className="w-full gap-2 h-9">
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Beaker className="w-4 h-4" />}
        {loading ? "Analyzing..." : "Analyze"}
      </Button>

      {input.trim() && (
        <div>
          <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">
            Queued ({input.split("\n").filter((s) => s.trim()).length})
          </p>
          <div className="flex flex-col gap-1">
            {input
              .split("\n")
              .map((s) => s.trim())
              .filter(Boolean)
              .map((smi, i) => (
                <p
                  key={i}
                  className="font-mono text-[11px] text-muted-foreground truncate bg-background/60 rounded-md px-2.5 py-1.5 border border-border"
                >
                  {smi}
                </p>
              ))}
          </div>
        </div>
      )}

      <Separator />

      <div className="text-[11px] text-muted-foreground leading-relaxed">
        <p className="font-semibold text-foreground mb-2 text-xs">LogS Reference</p>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2.5 bg-green-400/5 border border-green-400/20 rounded-md px-2.5 py-1.5">
            <span className="w-2 h-2 rounded-full bg-green-400 shrink-0" />
            <span className="text-green-400/90 font-medium">&gt;= -1</span>
            <span className="ml-auto">Highly soluble</span>
          </div>
          <div className="flex items-center gap-2.5 bg-yellow-400/5 border border-yellow-400/20 rounded-md px-2.5 py-1.5">
            <span className="w-2 h-2 rounded-full bg-yellow-400 shrink-0" />
            <span className="text-yellow-400/90 font-medium">-1 to -3</span>
            <span className="ml-auto">Moderate</span>
          </div>
          <div className="flex items-center gap-2.5 bg-red-400/5 border border-red-400/20 rounded-md px-2.5 py-1.5">
            <span className="w-2 h-2 rounded-full bg-red-400 shrink-0" />
            <span className="text-red-400/90 font-medium">&lt; -3</span>
            <span className="ml-auto">Poorly soluble</span>
          </div>
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground/50 text-center pb-2">
        Press <kbd className="bg-muted px-1 py-0.5 rounded text-[10px] font-mono">B</kbd> to toggle sidebar
      </p>
    </div>
  );
}

// ---- Molecule detail card ----

function MoleculeDetailCard({ result, index }: { result: MoleculeResult; index: number }) {
  const [open, setOpen] = useState(false);
  const d = result.details;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.35 }}
    >
      <Card className="bg-card border-border overflow-hidden">
        {/* Header row */}
        <CardHeader className="pb-0 pt-4 px-5">
          <div className="flex items-start justify-between gap-4">
            <div className="flex flex-col gap-1 min-w-0 flex-1">
              <p className="font-mono text-[11px] text-muted-foreground break-all leading-relaxed">
                {result.smiles}
              </p>
              {d?.iupac_name && (
                <p className="text-sm font-semibold leading-snug">{d.iupac_name}</p>
              )}
              {d?.common_names && d.common_names.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {d.common_names.slice(0, 3).map((n) => (
                    <Badge key={n} variant="secondary" className="text-[10px] px-1.5 py-0 h-4 font-normal">
                      {n}
                    </Badge>
                  ))}
                  {d.common_names.length > 3 && (
                    <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4 font-normal">
                      +{d.common_names.length - 3} more
                    </Badge>
                  )}
                </div>
              )}
            </div>
            {/* LogS chip */}
            <div className="shrink-0 flex flex-col items-end gap-1">
              <p className={`text-2xl font-bold font-mono tabular-nums ${logSColor(result.logS)}`}>
                {result.logS ?? "---"}
              </p>
              <Badge
                variant="outline"
                className={`text-[10px] px-2 py-0 h-5 font-medium border-current ${logSColor(result.logS)}`}
              >
                {logSLabel(result.logS)}
              </Badge>
            </div>
          </div>

          {/* Descriptor strip */}
          {d && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-border/60">
              {[
                { label: "Formula", value: d.formula },
                { label: "Mol. Weight", value: `${d.mol_weight} g/mol` },
                { label: "LogP", value: d.logP },
                { label: "TPSA", value: `${d.tpsa} Å²` },
              ].map(({ label, value }) => (
                <div key={label} className="flex flex-col gap-0.5">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-widest">{label}</p>
                  <p className="text-xs font-semibold font-mono">{value}</p>
                </div>
              ))}
            </div>
          )}
        </CardHeader>

        {d && (
          <>
            <button
              onClick={() => setOpen((v) => !v)}
              className="w-full px-5 py-2.5 mt-2 flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors border-t border-border/50"
            >
              {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              {open ? "Hide" : "Show"} full descriptor set
              <span className="ml-auto text-[10px] opacity-50">9 fields</span>
            </button>

            <AnimatePresence>
              {open && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="overflow-hidden"
                >
                  <CardContent className="px-5 pt-4 pb-5">
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-5 gap-y-3">
                      {[
                        ["H-Bond Donors", d.h_donors],
                        ["H-Bond Acceptors", d.h_acceptors],
                        ["Rotatable Bonds", d.rotatable_bonds],
                        ["Rings", d.rings],
                        ["Aromatic Rings", d.aromatic_rings],
                        ["Atom Count", d.atom_count],
                        ["Heavy Atoms", d.heavy_atom_count],
                        ["InChIKey", d.inchikey],
                        ["Canonical SMILES", d.canonical_smiles],
                      ].map(([label, value]) => (
                        <div key={String(label)} className="flex flex-col gap-0.5">
                          <span className="text-[10px] text-muted-foreground uppercase tracking-widest">
                            {label}
                          </span>
                          <span className="font-mono text-xs break-all leading-relaxed">{value}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}

        {result.error && (
          <CardContent className="px-5 pb-4 pt-0">
            <div className="flex items-center gap-2 text-destructive text-xs bg-destructive/5 border border-destructive/20 rounded-md px-3 py-2">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              {result.error}
            </div>
          </CardContent>
        )}
      </Card>
    </motion.div>
  );
}

// ---- Loading skeleton ----

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      {[1, 2, 3].map((i) => (
        <Card key={i} className="bg-card border-border">
          <CardHeader>
            <Skeleton className="h-3 w-32 mb-2" />
            <Skeleton className="h-5 w-48 mb-1" />
            <div className="grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-border">
              {[1, 2, 3, 4].map((j) => (
                <div key={j}>
                  <Skeleton className="h-2.5 w-12 mb-1" />
                  <Skeleton className="h-3.5 w-16" />
                </div>
              ))}
            </div>
          </CardHeader>
        </Card>
      ))}
    </div>
  );
}

// ---- Section header ----

function SectionHeader({ title, desc }: { title: string; desc?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
    >
      <h2 className="text-sm font-semibold">{title}</h2>
      {desc && <p className="text-xs text-muted-foreground mt-0.5">{desc}</p>}
    </motion.div>
  );
}

// ==== MAIN PAGE ====

export default function DashboardPage() {
  const router = useRouter();
  const [input, setInput] = useState(DEFAULT_SMILES);
  const [results, setResults] = useState<MoleculeResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // User
  const [userId, setUserId] = useState<string | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);

  // History
  const [historyOpen, setHistoryOpen] = useState(false);
  const [history, setHistory] = useState<AnalyticsRecord[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    setUserId(localStorage.getItem("user_id"));
    setUserEmail(localStorage.getItem("user_email"));
  }, []);

  // ⌨️  Press B to toggle sidebar
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (
        e.key.toLowerCase() === "b" &&
        !e.metaKey && !e.ctrlKey && !e.altKey &&
        (e.target as HTMLElement).tagName !== "INPUT" &&
        (e.target as HTMLElement).tagName !== "TEXTAREA"
      ) {
        setSidebarOpen((v) => !v);
      }
    },
    []
  );
  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  async function loadHistory() {
    if (!userId) return;
    setHistoryLoading(true);
    try {
      const data = await getAnalyticsByUser(userId);
      setHistory(data);
    } catch {
      toast.error("Could not load history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  function openHistory() {
    setHistoryOpen(true);
    loadHistory();
  }

  function deleteHistoryItem(id: string) {
    deleteAnalytics(id)
      .then(() => {
        setHistory((prev) => prev.filter((r) => r.id !== id));
        toast.success("Record deleted.");
      })
      .catch(() => toast.error("Failed to delete record."));
  }

  function loadFromHistory(smiles: string) {
    setInput(smiles);
    setHistoryOpen(false);
    toast.success("SMILES loaded into editor.");
  }

  function handleSignOut() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user_id");
    localStorage.removeItem("user_email");
    router.push("/auth/signin");
  }

  const userInitial = userEmail ? userEmail[0].toUpperCase() : "?";

  async function handleAnalyze() {
    const smilesList = input
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);

    if (smilesList.length === 0) {
      toast.error("Please enter at least one SMILES string.");
      return;
    }

    setLoading(true);
    setResults(null);
    try {
      const data = await analyzeSmiles(smilesList);
      setResults(data);
      toast.success(`Analyzed ${data.length} molecule${data.length > 1 ? "s" : ""}`);
      // Persist each molecule to analytics (fire-and-forget)
      if (userId) {
        data.forEach((r) => {
          if (!r.error) {
            createAnalytics({
              user_id: userId,
              user_email: userEmail,
              smiles: r.smiles,
              logS: r.logS,
              details: r.details,
            }).catch(() => {/* silent fail */});
          }
        });
      }
    } catch {
      toast.error("Failed to connect to API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  const chartData = results
    ?.filter((r) => r.logS !== null)
    .map((r, i) => ({
      name: r.details?.formula ?? `M${i + 1}`,
      logS: r.logS,
    }));

  return (
    <div className="min-h-screen flex bg-background">
      {/* ==== Desktop Sidebar ==== */}
      <motion.aside
        animate={{ width: sidebarOpen ? 288 : 56 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="hidden md:flex h-screen sticky top-0 shrink-0 border-r border-border flex-col bg-card overflow-x-hidden"
      >
        {/* Sidebar header */}
        <div className="px-4 py-4 border-b border-border flex items-center gap-2 min-h-[57px]">
          <Logo width={24} height={24} className="shrink-0" />
          <AnimatePresence>
            {sidebarOpen && (
              <motion.span
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                className="font-semibold tracking-tight text-sm whitespace-nowrap overflow-hidden"
              >
                MolSol
              </motion.span>
            )}
          </AnimatePresence>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen((v) => !v)}
            className="ml-auto h-8 w-8 shrink-0 text-muted-foreground"
            title={sidebarOpen ? "Collapse sidebar (B)" : "Expand sidebar (B)"}
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-4 h-4" />
            ) : (
              <PanelLeftOpen className="w-4 h-4" />
            )}
          </Button>
        </div>

        {/* Sidebar body */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden"
            >
              <SidebarBody
                input={input}
                setInput={setInput}
                loading={loading}
                onAnalyze={handleAnalyze}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.aside>

      {/* ==== Main area ==== */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Topbar */}
        <div className="border-b border-border px-4 md:px-8 py-3 flex items-center gap-3">
          {/* Mobile menu */}
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="md:hidden shrink-0">
                <Menu className="w-4 h-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-72 p-0 flex flex-col">
              <SheetHeader className="px-5 py-4 border-b border-border shrink-0">
                <div className="flex items-center gap-2">
                  <Logo width={24} height={24} />
                  <SheetTitle className="text-sm font-semibold tracking-tight">MolSol</SheetTitle>
                </div>
              </SheetHeader>
              <div className="flex-1 overflow-y-auto">
                <SidebarBody
                  input={input}
                  setInput={setInput}
                  loading={loading}
                  onAnalyze={handleAnalyze}
                />
              </div>
            </SheetContent>
          </Sheet>

          <Atom className="w-4 h-4 text-muted-foreground shrink-0 hidden sm:block" />
          <div className="flex-1 min-w-0">
            <h1 className="text-sm font-semibold truncate">Molecular Solubility Dashboard</h1>
            <p className="text-xs text-muted-foreground hidden sm:block">
              Predict and analyze solubility from SMILES notation
            </p>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            {/* History */}
            {userId && (
              <Button variant="ghost" size="icon" onClick={openHistory} title="View history">
                <History className="w-4 h-4" />
              </Button>
            )}

            <ThemeToggle />

            {/* User avatar */}
            {userId ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                    <Avatar className="w-7 h-7">
                      <AvatarFallback className="text-xs bg-primary text-primary-foreground">
                        {userInitial}
                      </AvatarFallback>
                    </Avatar>
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-60">
                  <DropdownMenuLabel className="flex flex-col gap-0.5 pb-2">
                    <span className="text-xs font-normal text-muted-foreground">Signed in as</span>
                    <span className="text-sm font-medium truncate">{userEmail ?? "User"}</span>
                    {userId && (
                      <span className="font-mono text-[10px] text-muted-foreground truncate">
                        {userId.slice(0, 8)}…
                      </span>
                    )}
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={openHistory} className="gap-2 cursor-pointer">
                    <Clock className="w-3.5 h-3.5" />
                    View History
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    onClick={handleSignOut}
                    className="gap-2 cursor-pointer text-destructive focus:text-destructive"
                  >
                    <LogOut className="w-3.5 h-3.5" />
                    Sign Out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button asChild size="sm" variant="outline" className="text-xs h-7 px-3">
                <Link href="/auth/signin">
                  <User className="w-3 h-3 mr-1" />
                  Sign In
                </Link>
              </Button>
            )}
          </div>
        </div>

        {/* History Sheet */}
        <Sheet open={historyOpen} onOpenChange={setHistoryOpen}>
          <SheetContent side="right" className="w-full sm:w-[480px] p-0 flex flex-col overflow-hidden h-full">
            <SheetHeader className="px-5 py-4 border-b border-border shrink-0">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-muted-foreground" />
                <SheetTitle className="text-sm font-semibold">Analysis History</SheetTitle>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                {history.length} previous session{history.length !== 1 ? "s" : ""}
              </p>
            </SheetHeader>

            <ScrollArea className="flex-1 min-h-0">
              <div className="p-4 flex flex-col gap-3">
                {historyLoading && (
                  <div className="flex flex-col gap-3">
                    {[1, 2, 3].map((i) => (
                      <Card key={i} className="bg-card border-border">
                        <CardHeader className="pb-3">
                          <Skeleton className="h-3 w-24 mb-2" />
                          <Skeleton className="h-4 w-40" />
                          <div className="grid grid-cols-3 gap-2 mt-3 pt-3 border-t border-border">
                            {[1, 2, 3].map((j) => <Skeleton key={j} className="h-6" />)}
                          </div>
                        </CardHeader>
                      </Card>
                    ))}
                  </div>
                )}

                {!historyLoading && !userId && (
                  <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
                    <User className="w-8 h-8 text-muted-foreground" />
                    <p className="text-sm font-medium">Not signed in</p>
                    <p className="text-xs text-muted-foreground">Sign in to save and view history.</p>
                    <Button asChild size="sm" variant="outline">
                      <Link href="/auth/signin">Sign In</Link>
                    </Button>
                  </div>
                )}

                {!historyLoading && userId && history.length === 0 && (
                  <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
                    <FlaskConical className="w-8 h-8 text-muted-foreground" />
                    <p className="text-sm font-medium">No history yet</p>
                    <p className="text-xs text-muted-foreground">Analyses you run will appear here.</p>
                  </div>
                )}

                {!historyLoading &&
                  history.map((row) => (
                    <Card key={row.id} className="bg-card border-border hover:border-border/80 transition-colors">
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex flex-col gap-0.5 min-w-0">
                            <p className="font-mono text-[11px] text-muted-foreground break-all leading-relaxed">
                              {row.smiles}
                            </p>
                            {row.iupac_name && (
                              <p className="text-xs font-semibold">{row.iupac_name}</p>
                            )}
                          </div>
                          <div className="text-right shrink-0 flex flex-col items-end gap-0.5">
                            <p className={`text-base font-bold font-mono ${logSColor(row.logS)}`}>
                              {row.logS ?? "---"}
                            </p>
                            <Badge variant="outline" className={`text-[9px] px-1.5 py-0 h-4 ${logSColor(row.logS)}`}>
                              {logSLabel(row.logS)}
                            </Badge>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-2 mt-2 pt-2 border-t border-border">
                          {[
                            ["Formula", row.formula],
                            ["MW", row.mol_weight ? `${row.mol_weight}` : "—"],
                            ["LogP", row.logP ?? "—"],
                          ].map(([label, value]) => (
                            <div key={String(label)}>
                              <p className="text-[10px] text-muted-foreground uppercase tracking-wide">{label}</p>
                              <p className="text-xs font-mono font-medium">{value}</p>
                            </div>
                          ))}
                        </div>

                        <div className="flex items-center justify-between mt-2 pt-2 border-t border-border">
                          <p className="text-[10px] text-muted-foreground">
                            {new Date(row.created_at).toLocaleString()}
                          </p>
                          <div className="flex items-center gap-1">
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-6 w-6"
                              title="Load into editor"
                              onClick={() => loadFromHistory(row.smiles)}
                            >
                              <RotateCcw className="w-3 h-3" />
                            </Button>
                            <Button
                              size="icon"
                              variant="ghost"
                              className="h-6 w-6 text-destructive hover:text-destructive"
                              title="Delete record"
                              onClick={() => deleteHistoryItem(row.id)}
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </div>
                        </div>
                      </CardHeader>
                    </Card>
                  ))}
              </div>
            </ScrollArea>
          </SheetContent>
        </Sheet>

        {/* Content */}
        <main className="flex-1 overflow-y-auto">
          <div className="px-4 md:px-8 py-6 md:py-8 flex flex-col gap-8 md:gap-10 max-w-5xl">
            {/* Empty state */}
            {!results && !loading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
                className="flex flex-col items-center justify-center py-24 sm:py-32 text-center gap-4"
              >
                <div className="w-12 h-12 rounded-full border border-border flex items-center justify-center">
                  <FlaskConical className="w-5 h-5 text-muted-foreground" />
                </div>
                <div>
                  <p className="font-medium">No analysis yet</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Enter SMILES in the sidebar and click Analyze
                  </p>
                </div>
              </motion.div>
            )}

            {/* Loading */}
            {loading && <LoadingSkeleton />}

            {/* Results */}
            {results && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4 }}
                className="flex flex-col gap-8 md:gap-10"
              >
                {/* Section 1 -- Input SMILES */}
                <section>
                  <SectionHeader
                    title="Input SMILES"
                    desc={`${results.length} molecule${results.length > 1 ? "s" : ""} submitted`}
                  />
                  <div className="flex flex-wrap gap-2 mt-3">
                    {results.map((r, i) => (
                      <Badge key={i} variant="outline" className="font-mono text-xs">
                        {r.smiles}
                      </Badge>
                    ))}
                  </div>
                </section>

                <Separator />

                {/* Section 2 -- Descriptors Table */}
                <section>
                  <SectionHeader
                    title="Computed Molecular Descriptors"
                    desc="Key physicochemical properties used by the model and reported by PubChem"
                  />
                  <div className="mt-3 rounded-lg border border-border overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-muted/30">
                          <TableHead className="font-mono text-xs">SMILES</TableHead>
                          <TableHead className="text-xs text-right">MolLogP</TableHead>
                          <TableHead className="text-xs text-right">MolWt</TableHead>
                          <TableHead className="text-xs text-right">Rot. Bonds</TableHead>
                          <TableHead className="text-xs text-right">TPSA</TableHead>
                          <TableHead className="text-xs text-right">H-Don</TableHead>
                          <TableHead className="text-xs text-right">H-Acc</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {results.map((r, i) =>
                          r.details ? (
                            <TableRow key={i}>
                              <TableCell className="font-mono text-xs text-muted-foreground max-w-[140px] truncate">
                                {r.smiles}
                              </TableCell>
                              <TableCell className="text-right font-mono text-xs">{r.details.logP}</TableCell>
                              <TableCell className="text-right font-mono text-xs">{r.details.mol_weight}</TableCell>
                              <TableCell className="text-right font-mono text-xs">{r.details.rotatable_bonds}</TableCell>
                              <TableCell className="text-right font-mono text-xs">{r.details.tpsa}</TableCell>
                              <TableCell className="text-right font-mono text-xs">{r.details.h_donors}</TableCell>
                              <TableCell className="text-right font-mono text-xs">{r.details.h_acceptors}</TableCell>
                            </TableRow>
                          ) : (
                            <TableRow key={i}>
                              <TableCell className="font-mono text-xs">{r.smiles}</TableCell>
                              <TableCell colSpan={6} className="text-xs text-muted-foreground text-center">
                                Details unavailable
                              </TableCell>
                            </TableRow>
                          )
                        )}
                      </TableBody>
                    </Table>
                  </div>
                </section>

                <Separator />

                {/* Section 3 -- Predicted LogS */}
                <section>
                  <SectionHeader
                    title="Predicted LogS Values"
                    desc="Aqueous solubility predictions from the RandomForest model"
                  />
                  <div className="mt-3 rounded-lg border border-border overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-muted/30">
                          <TableHead className="text-xs">Molecule</TableHead>
                          <TableHead className="text-xs">SMILES</TableHead>
                          <TableHead className="text-xs text-right">Predicted LogS</TableHead>
                          <TableHead className="text-xs">Class</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {results.map((r, i) => (
                          <TableRow key={i}>
                            <TableCell className="text-xs font-medium">
                              {r.details?.formula ?? `Molecule ${i + 1}`}
                            </TableCell>
                            <TableCell className="font-mono text-xs text-muted-foreground">
                              {r.smiles}
                            </TableCell>
                            <TableCell className={`text-right font-mono text-sm font-bold ${logSColor(r.logS)}`}>
                              {r.logS ?? "---"}
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline" className={`text-[10px] ${logSColor(r.logS)}`}>
                                {logSLabel(r.logS)}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </section>

                <Separator />

                {/* Section 4 -- LogS Chart */}
                {chartData && chartData.length > 0 && (
                  <section>
                    <SectionHeader
                      title="LogS Visualization"
                      desc="Predicted solubility values across molecules"
                    />
                    <div className="mt-3 h-40 sm:h-52">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(128,128,128,0.15)" />
                          <XAxis
                            dataKey="name"
                            tick={{ fontSize: 11, fill: "#888" }}
                            axisLine={false}
                            tickLine={false}
                          />
                          <YAxis
                            tick={{ fontSize: 11, fill: "#888" }}
                            axisLine={false}
                            tickLine={false}
                          />
                          <Tooltip
                            contentStyle={{
                              background: "var(--color-card, #1a1a1a)",
                              border: "1px solid var(--color-border, #333)",
                              borderRadius: 8,
                              fontSize: 12,
                            }}
                            cursor={{ fill: "rgba(128,128,128,0.08)" }}
                          />
                          <ReferenceLine y={-1} stroke="#4ade80" strokeDasharray="4 4" strokeWidth={1} />
                          <ReferenceLine y={-3} stroke="#facc15" strokeDasharray="4 4" strokeWidth={1} />
                          <Bar dataKey="logS" fill="#a3a3a3" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="flex flex-wrap items-center gap-4 mt-2">
                      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                        <span className="w-6 border-t border-green-400 border-dashed block" /> &gt;= -1 Highly soluble
                      </div>
                      <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                        <span className="w-6 border-t border-yellow-400 border-dashed block" /> &gt;= -3 Moderate
                      </div>
                    </div>
                  </section>
                )}

                <Separator />

                {/* Section 5 -- Molecule Details */}
                <section>
                  <SectionHeader
                    title="Molecule Details"
                    desc="Per-molecule analysis including IUPAC names, common synonyms, and full descriptor set"
                  />
                  <div className="mt-4 flex flex-col gap-4">
                    {results.map((r, i) => (
                      <MoleculeDetailCard key={i} result={r} index={i} />
                    ))}
                  </div>
                </section>
              </motion.div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
