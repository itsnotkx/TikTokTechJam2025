import React, { createContext, useContext, useState, ReactNode } from "react";

type VideoContextType = {
  videoUri: string | null;
  setVideoUri: (uri: string | null) => void;
};

const VideoContext = createContext<VideoContextType | undefined>(undefined);

export function VideoProvider({ children }: { children: ReactNode }) {
  const [videoUri, setVideoUri] = useState<string | null>(null);

  return (
    <VideoContext.Provider value={{ videoUri, setVideoUri }}>
      {children}
    </VideoContext.Provider>
  );
}

// Custom hook for convenience
export function useVideo() {
  const context = useContext(VideoContext);
  if (!context) {
    throw new Error("useVideo must be used within a VideoProvider");
  }
  return context;
}