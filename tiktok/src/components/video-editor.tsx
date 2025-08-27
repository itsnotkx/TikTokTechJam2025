// import { useState, useRef, useEffect } from "@lynx-js/react"

// interface VideoEditorProps {
//   videoFile: File
//   onBack: () => void
// }

// interface FlaggedScene {
//   timestamp: number
//   type: "license_plate" | "credit_card" | "personal_info"
//   confidence: number
//   blurred: boolean
// }

// export default function VideoEditor({ videoFile, onBack }: VideoEditorProps) {
//   const videoRef = useRef<HTMLVideoElement>(null)
//   const [isPlaying, setIsPlaying] = useState(false)
//   const [currentTime, setCurrentTime] = useState(0)
//   const [duration, setDuration] = useState(0)
//   const [volume, setVolume] = useState(100)
//   const [isAnalyzing, setIsAnalyzing] = useState(false)
//   const [analysisComplete, setAnalysisComplete] = useState(false)
//   const [flaggedScenes, setFlaggedScenes] = useState<FlaggedScene[]>([])
//   const [videoUrl, setVideoUrl] = useState<string>("")

//   useEffect(() => {
//     const url = URL.createObjectURL(videoFile)
//     setVideoUrl(url)
//     return () => URL.revokeObjectURL(url)
//   }, [videoFile])

//   const togglePlay = () => {
//     if (videoRef.current) {
//       if (isPlaying) {
//         videoRef.current.pause()
//       } else {
//         videoRef.current.play()
//       }
//       setIsPlaying(!isPlaying)
//     }
//   }

//   const handleTimeUpdate = () => {
//     if (videoRef.current) {
//       setCurrentTime(videoRef.current.currentTime)
//     }
//   }

//   const handleLoadedMetadata = () => {
//     if (videoRef.current) {
//       setDuration(videoRef.current.duration)
//     }
//   }

//   const handleSeek = (value: number) => {
//     if (videoRef.current) {
//       videoRef.current.currentTime = value
//       setCurrentTime(value)
//     }
//   }

//   const handleVolumeChange = (value: number) => {
//     setVolume(value)
//     if (videoRef.current) {
//       videoRef.current.volume = value / 100
//     }
//   }

//   const startAIAnalysis = () => {
//     setIsAnalyzing(true)
//     // Simulate AI analysis
//     setTimeout(() => {
//       const mockFlaggedScenes: FlaggedScene[] = [
//         { timestamp: 15.5, type: "license_plate", confidence: 0.92, blurred: false },
//         { timestamp: 42.3, type: "credit_card", confidence: 0.87, blurred: false },
//         { timestamp: 78.1, type: "personal_info", confidence: 0.94, blurred: false },
//       ]
//       setFlaggedScenes(mockFlaggedScenes)
//       setIsAnalyzing(false)
//       setAnalysisComplete(true)
//     }, 3000)
//   }

//   const toggleBlur = (timestamp: number) => {
//     setFlaggedScenes((scenes) =>
//       scenes.map((scene) => (scene.timestamp === timestamp ? { ...scene, blurred: !scene.blurred } : scene)),
//     )
//   }

//   const jumpToScene = (timestamp: number) => {
//     if (videoRef.current) {
//       videoRef.current.currentTime = timestamp
//       setCurrentTime(timestamp)
//     }
//   }

//   const formatTime = (time: number) => {
//     const minutes = Math.floor(time / 60)
//     const seconds = Math.floor(time % 60)
//     return `${minutes}:${seconds.toString().padStart(2, "0")}`
//   }

//   const getSceneTypeIcon = (type: string) => {
//     switch (type) {
//       case "license_plate":
//         return "🚗"
//       case "credit_card":
//         return "💳"
//       case "personal_info":
//         return "📄"
//       default:
//         return "⚠️"
//     }
//   }

//   const getSceneTypeLabel = (type: string) => {
//     switch (type) {
//       case "license_plate":
//         return "License Plate"
//       case "credit_card":
//         return "Credit Card"
//       case "personal_info":
//         return "Personal Info"
//       default:
//         return "Sensitive Content"
//     }
//   }

//   return (
//     <view style={{ minHeight: "100vh", backgroundColor: "#0a0a0a", color: "#ffffff" }}>
//       {/* Header */}
//       <view style={{ borderBottomWidth: 1, borderBottomColor: "#27272a", padding: 16 }}>
//         <view
//           style={{
//             flexDirection: "row",
//             alignItems: "center",
//             justifyContent: "space-between",
//             maxWidth: 1400,
//             margin: "auto",
//           }}
//         >
//           <view style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
//             <view
//               style={{
//                 backgroundColor: "#18181b",
//                 padding: 12,
//                 borderRadius: 6,
//                 cursor: "pointer",
//               }}
//               onPress={onBack}
//             >
//               <text style={{ color: "#ffffff" }}>← Back</text>
//             </view>
//             <text style={{ fontSize: 20, fontWeight: "600" }}>Video Editor</text>
//           </view>

//           <view style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
//             <view
//               style={{
//                 backgroundColor: "#18181b",
//                 borderWidth: 1,
//                 borderColor: "#3f3f46",
//                 padding: 12,
//                 borderRadius: 6,
//                 cursor: "pointer",
//               }}
//             >
//               <text style={{ color: "#ffffff" }}>📥 Export</text>
//             </view>
//           </view>
//         </view>
//       </view>

//       <view style={{ flexDirection: "row", height: "calc(100vh - 73px)" }}>
//         {/* Main Editor */}
//         <view style={{ flex: 1, flexDirection: "column" }}>
//           {/* Video Preview */}
//           <view
//             style={{
//               flex: 1,
//               alignItems: "center",
//               justifyContent: "center",
//               backgroundColor: "rgba(0,0,0,0.2)",
//               padding: 32,
//             }}
//           >
//             <view style={{ position: "relative", maxWidth: 900, width: "100%" }}>
//               <video
//                 ref={videoRef}
//                 src={videoUrl}
//                 style={{ width: "100%", height: "auto", borderRadius: 8 }}
//                 onTimeUpdate={handleTimeUpdate}
//                 onLoadedMetadata={handleLoadedMetadata}
//               />

//               {/* Flagged Scene Overlays */}
//               {flaggedScenes.map((scene) => {
//                 const isCurrentScene = Math.abs(currentTime - scene.timestamp) < 1
//                 if (isCurrentScene && scene.blurred) {
//                   return (
//                     <view
//                       key={scene.timestamp}
//                       style={{
//                         position: "absolute",
//                         top: 0,
//                         left: 0,
//                         right: 0,
//                         bottom: 0,
//                         backgroundColor: "rgba(0,0,0,0.8)",
//                         borderRadius: 8,
//                         alignItems: "center",
//                         justifyContent: "center",
//                       }}
//                     >
//                       <view style={{ alignItems: "center" }}>
//                         <text style={{ fontSize: 48, marginBottom: 8 }}>🙈</text>
//                         <text style={{ fontSize: 18, fontWeight: "600", color: "white" }}>
//                           Sensitive Content Blurred
//                         </text>
//                         <text style={{ fontSize: 14, color: "#a1a1aa" }}>{getSceneTypeLabel(scene.type)}</text>
//                       </view>
//                     </view>
//                   )
//                 }
//                 return null
//               })}
//             </view>
//           </view>

//           {/* Timeline and Controls */}
//           <view style={{ borderTopWidth: 1, borderTopColor: "#27272a", padding: 24, gap: 16 }}>
//             {/* Timeline */}
//             <view style={{ position: "relative" }}>
//               <input
//                 type="range"
//                 min={0}
//                 max={duration}
//                 step={0.1}
//                 value={currentTime}
//                 onChange={(e) => handleSeek(Number.parseFloat(e.target.value))}
//                 style={{ width: "100%", height: 8, backgroundColor: "#3f3f46", borderRadius: 4 }}
//               />

//               {/* Flagged Scene Markers */}
//               {flaggedScenes.map((scene) => (
//                 <view
//                   key={scene.timestamp}
//                   style={{
//                     position: "absolute",
//                     top: 0,
//                     left: `${(scene.timestamp / duration) * 100}%`,
//                     width: 12,
//                     height: 12,
//                     backgroundColor: "#eab308",
//                     borderRadius: 6,
//                     transform: "translate(-50%, -50%)",
//                     cursor: "pointer",
//                   }}
//                   onPress={() => jumpToScene(scene.timestamp)}
//                 />
//               ))}
//             </view>

//             <view style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
//               <view style={{ flexDirection: "row", alignItems: "center", gap: 16 }}>
//                 {/* Playback Controls */}
//                 <view style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
//                   <view style={{ padding: 8, cursor: "pointer" }}>
//                     <text style={{ fontSize: 16 }}>⏮️</text>
//                   </view>
//                   <view
//                     style={{
//                       backgroundColor: "#8b5cf6",
//                       padding: 8,
//                       borderRadius: 6,
//                       cursor: "pointer",
//                     }}
//                     onPress={togglePlay}
//                   >
//                     <text style={{ fontSize: 16, color: "white" }}>{isPlaying ? "⏸️" : "▶️"}</text>
//                   </view>
//                   <view style={{ padding: 8, cursor: "pointer" }}>
//                     <text style={{ fontSize: 16 }}>⏭️</text>
//                   </view>
//                 </view>

//                 {/* Time Display */}
//                 <text style={{ fontSize: 14, color: "#a1a1aa" }}>
//                   {formatTime(currentTime)} / {formatTime(duration)}
//                 </text>

//                 {/* Volume */}
//                 <view style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
//                   <text style={{ fontSize: 16 }}>🔊</text>
//                   <input
//                     type="range"
//                     min={0}
//                     max={100}
//                     step={1}
//                     value={volume}
//                     onChange={(e) => handleVolumeChange(Number.parseInt(e.target.value))}
//                     style={{ width: 80 }}
//                   />
//                 </view>
//               </view>

//               {/* AI Analysis Button */}
//               <view
//                 style={{
//                   backgroundColor: isAnalyzing || analysisComplete ? "#3f3f46" : "#8b5cf6",
//                   paddingHorizontal: 16,
//                   paddingVertical: 12,
//                   borderRadius: 8,
//                   cursor: isAnalyzing || analysisComplete ? "not-allowed" : "pointer",
//                   opacity: isAnalyzing || analysisComplete ? 0.7 : 1,
//                 }}
//                 onPress={!isAnalyzing && !analysisComplete ? startAIAnalysis : undefined}
//               >
//                 <text style={{ color: "white", fontWeight: "600" }}>
//                   {isAnalyzing
//                     ? "🔄 Analyzing..."
//                     : analysisComplete
//                       ? "🛡️ Analysis Complete"
//                       : "🛡️ Analyze for Privacy Issues"}
//                 </text>
//               </view>
//             </view>
//           </view>
//         </view>

//         {/* Sidebar */}
//         <view style={{ width: 320, borderLeftWidth: 1, borderLeftColor: "#27272a", backgroundColor: "#18181b" }}>
//           <view style={{ padding: 16, gap: 16 }}>
//             {/* Editing Tools */}
//             <view style={{ backgroundColor: "#27272a", borderRadius: 8, padding: 16 }}>
//               <text style={{ fontWeight: "600", marginBottom: 12 }}>Editing Tools</text>
//               <view style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
//                 <view
//                   style={{
//                     backgroundColor: "#3f3f46",
//                     paddingHorizontal: 12,
//                     paddingVertical: 8,
//                     borderRadius: 6,
//                     flex: 1,
//                     minWidth: "45%",
//                   }}
//                 >
//                   <text style={{ fontSize: 12, color: "white", textAlign: "center" }}>📝 Text</text>
//                 </view>
//                 <view
//                   style={{
//                     backgroundColor: "#3f3f46",
//                     paddingHorizontal: 12,
//                     paddingVertical: 8,
//                     borderRadius: 6,
//                     flex: 1,
//                     minWidth: "45%",
//                   }}
//                 >
//                   <text style={{ fontSize: 12, color: "white", textAlign: "center" }}>🎵 Audio</text>
//                 </view>
//                 <view
//                   style={{
//                     backgroundColor: "#3f3f46",
//                     paddingHorizontal: 12,
//                     paddingVertical: 8,
//                     borderRadius: 6,
//                     flex: 1,
//                     minWidth: "45%",
//                   }}
//                 >
//                   <text style={{ fontSize: 12, color: "white", textAlign: "center" }}>✂️ Cut</text>
//                 </view>
//                 <view
//                   style={{
//                     backgroundColor: "#3f3f46",
//                     paddingHorizontal: 12,
//                     paddingVertical: 8,
//                     borderRadius: 6,
//                     flex: 1,
//                     minWidth: "45%",
//                   }}
//                 >
//                   <text style={{ fontSize: 12, color: "white", textAlign: "center" }}>✨ Effects</text>
//                 </view>
//               </view>
//             </view>

//             {/* AI Analysis Results */}
//             {analysisComplete && (
//               <view style={{ backgroundColor: "#27272a", borderRadius: 8, padding: 16 }}>
//                 <view style={{ flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 12 }}>
//                   <text style={{ fontSize: 16, color: "#eab308" }}>⚠️</text>
//                   <text style={{ fontWeight: "600" }}>Privacy Issues Found</text>
//                 </view>
//                 <view style={{ gap: 12 }}>
//                   {flaggedScenes.map((scene) => (
//                     <view
//                       key={scene.timestamp}
//                       style={{ borderWidth: 1, borderColor: "#3f3f46", borderRadius: 8, padding: 12 }}
//                     >
//                       <view
//                         style={{
//                           flexDirection: "row",
//                           alignItems: "center",
//                           justifyContent: "space-between",
//                           marginBottom: 8,
//                         }}
//                       >
//                         <view style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
//                           <text style={{ fontSize: 18 }}>{getSceneTypeIcon(scene.type)}</text>
//                           <view>
//                             <text style={{ fontSize: 14, fontWeight: "500" }}>{getSceneTypeLabel(scene.type)}</text>
//                             <text style={{ fontSize: 12, color: "#a1a1aa" }}>
//                               {formatTime(scene.timestamp)} • {Math.round(scene.confidence * 100)}% confidence
//                             </text>
//                           </view>
//                         </view>
//                       </view>

//                       <view style={{ flexDirection: "row", gap: 8 }}>
//                         <view
//                           style={{
//                             backgroundColor: "#3f3f46",
//                             paddingHorizontal: 12,
//                             paddingVertical: 6,
//                             borderRadius: 4,
//                             flex: 1,
//                             cursor: "pointer",
//                           }}
//                           onPress={() => jumpToScene(scene.timestamp)}
//                         >
//                           <text style={{ fontSize: 12, color: "white", textAlign: "center" }}>Jump to Scene</text>
//                         </view>
//                         <view
//                           style={{
//                             backgroundColor: scene.blurred ? "#16a34a" : "#8b5cf6",
//                             paddingHorizontal: 12,
//                             paddingVertical: 6,
//                             borderRadius: 4,
//                             cursor: "pointer",
//                           }}
//                           onPress={() => toggleBlur(scene.timestamp)}
//                         >
//                           <text style={{ fontSize: 12, color: "white" }}>
//                             {scene.blurred ? "🙈 Blurred" : "👁️ Blur"}
//                           </text>
//                         </view>
//                       </view>
//                     </view>
//                   ))}
//                 </view>
//               </view>
//             )}
//           </view>
//         </view>
//       </view>
//     </view>
//   )
// }
