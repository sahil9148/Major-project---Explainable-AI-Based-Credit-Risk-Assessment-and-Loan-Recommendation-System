"use client";

/**
 * CanvasScroll.tsx
 * -----------------
 * The signature "Apple-style" 3D scroll sequence for the landing page.
 *
 * A translucent glass credit card floats in a softly lit studio
 * environment. As the user scrolls through the pinned hero section, GSAP's
 * ScrollTrigger drives the card's rotation, position, and scale directly
 * on the underlying Three.js objects -- React Three Fiber's render loop
 * (which runs continuously by default) picks up those mutations on the
 * next frame automatically, so no extra state management is needed to
 * keep the animation in sync with scroll.
 *
 * This uses a standard Drei RoundedBox as a stand-in "card" geometry, as
 * requested. Swap it for an imported GLTF model (useGLTF from
 * @react-three/drei) once you have a real 3D asset -- every other part of
 * this file (lighting, materials, scroll rig, post-processing) keeps
 * working unchanged.
 *
 * Import this with `next/dynamic` and `{ ssr: false }` wherever it's used
 * (see app/page.tsx) -- WebGL has no meaning on the server, so there's no
 * reason to pay for a server render of this component.
 */

import { useEffect, useRef, type RefObject } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  Environment,
  Float,
  MeshTransmissionMaterial,
  RoundedBox,
  ContactShadows,
} from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

if (typeof window !== "undefined") {
  gsap.registerPlugin(ScrollTrigger);
}

interface GlassCardProps {
  groupRef: RefObject<THREE.Group>;
}

function GlassCard({ groupRef }: GlassCardProps) {
  const chipRef = useRef<THREE.Mesh>(null);

  // A slow, continuous idle spin on the chip only (a child of the group
  // GSAP controls) -- purely decorative, and doesn't fight with the
  // scroll-driven rotation since it's a different object.
  useFrame((_, delta) => {
    if (chipRef.current) {
      chipRef.current.rotation.z += delta * 0.15;
    }
  });

  return (
    <group ref={groupRef}>
      <Float speed={1.2} rotationIntensity={0.15} floatIntensity={0.4}>
        <RoundedBox args={[3.2, 2, 0.12]} radius={0.18} smoothness={6} castShadow>
          <MeshTransmissionMaterial
            thickness={0.6}
            roughness={0.06}
            transmission={1}
            ior={1.4}
            chromaticAberration={0.04}
            backside
            color="#dfeeff"
          />
        </RoundedBox>
        <mesh ref={chipRef} position={[-1.05, 0.55, 0.075]}>
          <boxGeometry args={[0.45, 0.35, 0.02]} />
          <meshStandardMaterial
            color="#ffd580"
            metalness={0.9}
            roughness={0.25}
            emissive="#ffb347"
            emissiveIntensity={0.15}
          />
        </mesh>
      </Float>
    </group>
  );
}

function Scene({ groupRef }: GlassCardProps) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <spotLight position={[5, 6, 5]} angle={0.35} penumbra={1} intensity={2.2} castShadow />
      <Environment preset="studio" />
      <GlassCard groupRef={groupRef} />
      <ContactShadows position={[0, -1.4, 0]} opacity={0.45} scale={10} blur={2.5} far={4} />
      <EffectComposer>
        <Bloom intensity={0.65} luminanceThreshold={0.35} luminanceSmoothing={0.9} mipmapBlur />
        <Vignette eskil={false} offset={0.15} darkness={0.6} />
      </EffectComposer>
    </>
  );
}

export default function CanvasScroll() {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion || !groupRef.current || !wrapperRef.current) return;

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: wrapperRef.current,
          start: "top top",
          end: "+=200%",
          scrub: 1,
          pin: true,
        },
      });

      tl.to(groupRef.current!.rotation, { y: Math.PI * 1.5, x: 0.3, ease: "none" }, 0)
        .to(groupRef.current!.position, { z: 1.5, y: -0.3, ease: "none" }, 0)
        .to(groupRef.current!.scale, { x: 1.6, y: 1.6, z: 1.6, ease: "none" }, 0.5);
    }, wrapperRef);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={wrapperRef} className="relative h-screen w-full overflow-hidden bg-black">
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [0, 0, 6], fov: 35 }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene groupRef={groupRef} />
      </Canvas>

      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
        <h1 className="text-6xl font-semibold tracking-tight text-white md:text-8xl">
          RiskLens
        </h1>
        <p className="mt-4 max-w-md px-6 text-lg text-white/60">
          Loan decisions you can see straight through.
        </p>
      </div>
    </div>
  );
}
