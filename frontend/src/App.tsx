import './App.css'
import { useState } from "@lynx-js/react"

export function App(props: {
  onRender?: () => void
}) {

  const [selectedVideo, setSelectedVideo] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  async function handleVideoUpload() {
      try {
        const res = await (lynx as any).chooseVideo({
          sourceType: ['album'], // pick from gallery
          compressed: true
        })

        // res.tempFilePath → path to the chosen video
        console.log('Video chosen:', res.tempFilePath)

        // Now upload to your backend
        // await this.uploadVideo(res.tempFilePath)

        setIsUploading(true);
        setTimeout(() => {
          setSelectedVideo(res.tempFilePath);
          setIsUploading(false);
        }, 1500);

      } catch (err) {
        console.error('Video selection failed:', err)
      }
    }

  return (
    <scroll-view className="Background" scroll-orientation="vertical">
      <view className="App">
        {/* Header */}
        <view>
          <view
            style={{ flexDirection: "row", alignItems: "center", justifyContent: "center"}}
          >
            <text style={{ fontSize: "32px", fontWeight: "bold" }}>🛡️ AI Privacy Video Editor</text>
          </view>

          <text style={{ fontSize: "20px", color: "#a1a1aa", maxWidth: "600px", marginTop: '8px', marginBottom: "16px" }}>
            Upload your video and let our AI detect and blur sensitive information like license plates, credit cards,
            and personal details before you share.
          </text>
        </view>

        {/* Upload Area */}
        <view
          style={{
            maxWidth: "600px",
            backgroundColor: "#18181b",
            borderRadius: "12px",
            padding: "32px",
            borderWidth: "1px",
            borderColor: "#27272a",
          }}
        >
          <view
            style={{
              borderWidth: "2px",
              borderStyle: "dashed",
              borderColor: "#3f3f46",
              borderRadius: "8px",
              padding: "48px",
              textAlign: "center",
            }}
          >
            {isUploading ? (
              <view style={{ alignItems: "center", gap: 16 }}>
                <view
                  style={{
                    width: "48px",
                    height: "48px",
                    borderWidth: "4px",
                    borderColor: "#8b5cf6",
                    borderTopColor: "transparent",
                    borderRadius: "24px",
                    // Note: Lynx may not support CSS animations, this would need native implementation
                  }}
                ></view>
                <text style={{ fontSize: "18px" }}>Processing your video...</text>
              </view>
            ) : (
              <view style={{ alignItems: "center", gap: "24px" }}>
                <view style={{ alignItems: "center" }}>
                  <text style={{ fontSize: "64px", color: "#71717a" }}>🎥</text>
                  {/* <text style={{ fontSize: "24px", color: "#8b5cf6", position: "absolute", top: "-4px", right: "-4px" }}>📤</text> */}
                </view>

                <view style={{ alignItems: "center", gap: "8px", marginBottom: '8px'}}>
                  <text style={{ fontSize: "20px", fontWeight: "600" }}>Upload Your Video</text>
                  <text style={{ color: "#a1a1aa", textAlign: 'center'}}>Choose your video file to get started</text>
                </view>

                <view style={{ alignItems: "center"}}>
                  {/* <input
                    type="file"
                    accept="video/*"
                    onChange={handleVideoUpload}
                    style={{ display: "none" }}
                    id="video-upload"
                  /> */}
                  <view
                    style={{
                      backgroundColor: "#8b5cf6",
                      padding: "24px",
                      borderRadius: "8px",
                      cursor: "pointer",
                    }}
                    bindtap={handleVideoUpload}
                    // onPress={() => {
                    //   const input = document.getElementById("video-upload") as HTMLInputElement
                    //   input?.click()
                    // }}
                  >
                    <text style={{ color: "white", fontWeight: "600"}}>📤 Choose Video File</text>
                  </view>

                  <text style={{ fontSize: '14px', color: "#a1a1aa", marginTop: '8px'}}>
                    Supports MP4, MOV, AVI, and other common video formats
                  </text>
                </view>
              </view>
            )}
          </view>
        </view>
      </view>
    </scroll-view>
  )
}



