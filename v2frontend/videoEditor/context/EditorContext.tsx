import React, { createContext, useContext, useState, ReactNode } from 'react';

interface EditOperation {
  id: string;
  type: 'trim' | 'filter' | 'text' | 'audio';
  startTime: number;
  endTime?: number;
  parameters: Record<string, any>;
}

interface EditorContextType {
  // Video state
  videoDuration: number;
  currentTime: number;
  isPlaying: boolean;
  
  // Editing state
  editOperations: EditOperation[];
  trimStart: number;
  trimEnd: number;
  
  // Methods
  setVideoDuration: (duration: number) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  addEditOperation: (operation: EditOperation) => void;
  removeEditOperation: (id: string) => void;
  setTrimPoints: (start: number, end: number) => void;
  clearEdits: () => void;
}

const EditorContext = createContext<EditorContextType | undefined>(undefined);

export const EditorProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [editOperations, setEditOperations] = useState<EditOperation[]>([]);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);

  const addEditOperation = (operation: EditOperation) => {
    setEditOperations(prev => [...prev, operation]);
  };

  const removeEditOperation = (id: string) => {
    setEditOperations(prev => prev.filter(op => op.id !== id));
  };

  const setTrimPoints = (start: number, end: number) => {
    setTrimStart(start);
    setTrimEnd(end);
    
    // Add/update trim operation
    const trimOperation: EditOperation = {
      id: 'trim-main',
      type: 'trim',
      startTime: start,
      endTime: end,
      parameters: {}
    };
    
    setEditOperations(prev => {
      const withoutTrim = prev.filter(op => op.id !== 'trim-main');
      return [...withoutTrim, trimOperation];
    });
  };

  const clearEdits = () => {
    setEditOperations([]);
    setTrimStart(0);
    setTrimEnd(videoDuration);
  };

  return (
    <EditorContext.Provider value={{
      videoDuration,
      currentTime,
      isPlaying,
      editOperations,
      trimStart,
      trimEnd,
      setVideoDuration,
      setCurrentTime,
      setIsPlaying,
      addEditOperation,
      removeEditOperation,
      setTrimPoints,
      clearEdits
    }}>
      {children}
    </EditorContext.Provider>
  );
};

export const useEditor = () => {
  const context = useContext(EditorContext);
  if (!context) {
    throw new Error('useEditor must be used within EditorProvider');
  }
  return context;
};