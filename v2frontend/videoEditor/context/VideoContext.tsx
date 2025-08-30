import React, { createContext, useContext, useState, ReactNode } from "react";

type UploadStatus =
  | { status: "idle" }
  | { status: "uploading" }
  | { status: "processing"; jobId: string }
  | { status: "done"; jobId: string; result?: any }
  | { status: "error"; message: string };

type VideoContextType = {
  videoUri: string | null;
  setVideoUri: (uri: string | null) => void;

  upload: UploadStatus;
  setUpload: (u: UploadStatus) => void;
};

const VideoContext = createContext<VideoContextType | undefined>(undefined);

export function VideoProvider({ children }: { children: ReactNode }) {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadStatus>({ status: "idle" });

  return (
    <VideoContext.Provider value={{ videoUri, setVideoUri, upload, setUpload }}>
      {children}
    </VideoContext.Provider>
  );
}

// Custom hook
export function useVideo() {
  const context = useContext(VideoContext);
  if (!context) {
    throw new Error("useVideo must be used within a VideoProvider");
  }
  return context;
}
