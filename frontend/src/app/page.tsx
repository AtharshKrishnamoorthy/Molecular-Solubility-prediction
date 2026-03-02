"use client";

import Link from "next/link";
import { ArrowRight, Atom, Activity, FlaskConical } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { DotBackground } from "@/components/dot-background";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme-toggle";

const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" },
  }),
};

const features = [
  {
    icon: <Atom className="w-5 h-5" />,
    title: "SMILES Input",
    desc: "Enter one or multiple SMILES strings and analyze them all at once.",
  },
  {
    icon: <Activity className="w-5 h-5" />,
    title: "LogS Prediction",
    desc: "Get predicted aqueous solubility values from our trained ML model.",
  },
  {
    icon: <FlaskConical className="w-5 h-5" />,
    title: "Molecular Details",
    desc: "IUPAC name, formula, MW, LogP, H-donors, TPSA, and more via PubChem.",
  },
];

export default function Home() {
  return (
    <DotBackground>
      <main className="min-h-screen flex flex-col">
        {/* Nav */}
        <motion.nav
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="border-b border-border px-4 sm:px-6 py-4 flex items-center justify-between backdrop-blur-sm bg-background/80 sticky top-0 z-50"
        >
          <div className="flex items-center gap-2.5">
            <Logo width={26} height={26} />
            <span className="font-semibold tracking-tight text-sm">MolSol</span>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button asChild size="sm" variant="ghost" className="hidden sm:inline-flex">
              <Link href="/auth/signin">Sign In</Link>
            </Button>
            <Button asChild size="sm">
              <Link href="/auth/signup">Sign Up</Link>
            </Button>
          </div>
        </motion.nav>

        {/* Hero */}
        <section className="flex flex-1 flex-col items-center justify-center text-center px-4 sm:px-6 py-16 sm:py-24 gap-5">
          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={0}
          >
            <Badge variant="outline" className="text-xs tracking-widest uppercase font-mono">
              ML - RDKit - FastAPI
            </Badge>
          </motion.div>

          <motion.h1
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={1}
            className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight leading-tight max-w-2xl"
          >
            Molecular Solubility
            <br />
            <span className="text-muted-foreground">Prediction</span>
          </motion.h1>

          <motion.p
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={2}
            className="text-muted-foreground text-sm sm:text-base max-w-md leading-relaxed"
          >
            Predict the{" "}
            <span className="text-foreground font-medium">LogS solubility</span>{" "}
            of any molecule from its SMILES notation. Powered by a RandomForest
            model trained on the Delaney dataset.
          </motion.p>

          <motion.div
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            custom={3}
            className="flex flex-wrap justify-center gap-3 mt-2"
          >
            <Button asChild size="lg" className="gap-2">
              <Link href="/auth/signup">
                Get Started <ArrowRight className="w-4 h-4" />
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/dashboard">Try Without Account</Link>
            </Button>
          </motion.div>
        </section>

        <Separator />

        {/* Feature strip */}
        <section className="grid grid-cols-1 sm:grid-cols-3 divide-y sm:divide-y-0 sm:divide-x divide-border">
          {features.map(({ icon, title, desc }, i) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
              whileHover={{ y: -2 }}
              className="flex flex-col gap-2 px-6 sm:px-8 py-7 sm:py-8"
            >
              <div className="text-muted-foreground">{icon}</div>
              <p className="font-semibold text-sm">{title}</p>
              <p className="text-muted-foreground text-xs sm:text-sm leading-relaxed">{desc}</p>
            </motion.div>
          ))}
        </section>

        {/* Footer */}
        <footer className="border-t border-border px-4 sm:px-6 py-4 text-center text-xs text-muted-foreground">
          Built with Next.js, FastAPI, RDKit, and scikit-learn
        </footer>
      </main>
    </DotBackground>
  );
}
