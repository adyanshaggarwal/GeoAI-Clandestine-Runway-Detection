import type React from "react"
import type { Metadata } from "next"
import { Work_Sans, Open_Sans } from "next/font/google"
import "./globals.css"

export const metadata: Metadata = {
  title: "SkyWatch AI - Elevate Your Insights with AI-Powered Precision",
  description:
    "Transform runway detection into actionable intelligence. Harness the power of satellite imagery with ease using cutting-edge YOLO deep learning technology.",
  generator: "v0.app",
}

const workSans = Work_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-work-sans",
  weight: ["400", "600", "700"],
})

const openSans = Open_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-open-sans",
  weight: ["400", "600"],
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${workSans.variable} ${openSans.variable} antialiased`}>{children}</body>
    </html>
  )
}
