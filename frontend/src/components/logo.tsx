"use client";

import Image from "next/image";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

interface LogoProps {
  width?: number;
  height?: number;
  className?: string;
}

export function Logo({ width = 28, height = 28, className }: LogoProps) {
  const { theme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);
  if (!mounted) return <div style={{ width, height }} />;

  return (
    <Image
      src={theme === "dark" ? "/logos/dark-logo.png" : "/logos/light-logo.png"}
      alt="MolSol"
      width={width}
      height={height}
      className={className}
      priority
    />
  );
}
