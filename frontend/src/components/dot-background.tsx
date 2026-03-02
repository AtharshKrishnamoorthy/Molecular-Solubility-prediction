"use client";

import {
  useMotionTemplate,
  useMotionValue,
  useSpring,
  motion,
} from "framer-motion";

export function DotBackground({ children }: { children?: React.ReactNode }) {
  const mouseX = useMotionValue(-1000);
  const mouseY = useMotionValue(-1000);

  const springX = useSpring(mouseX, { stiffness: 90, damping: 22 });
  const springY = useSpring(mouseY, { stiffness: 90, damping: 22 });

  const spotlight = useMotionTemplate`radial-gradient(520px circle at ${springX}px ${springY}px, rgba(255,255,255,0.07), transparent 50%)`;

  return (
    <div
      className="relative overflow-hidden"
      onMouseMove={(e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        mouseX.set(e.clientX - rect.left);
        mouseY.set(e.clientY - rect.top);
      }}
      onMouseLeave={() => {
        mouseX.set(-1000);
        mouseY.set(-1000);
      }}
    >
      {/* Static dot grid */}
      <div
        className="pointer-events-none absolute inset-0 dark:[background-image:radial-gradient(rgba(255,255,255,0.08)_1px,transparent_1px)] [background-image:radial-gradient(rgba(0,0,0,0.08)_1px,transparent_1px)] [background-size:26px_26px]"
        aria-hidden
      />
      {/* Mouse spotlight glow */}
      <motion.div
        className="pointer-events-none absolute inset-0"
        style={{ background: spotlight }}
        aria-hidden
      />
      {/* Content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
