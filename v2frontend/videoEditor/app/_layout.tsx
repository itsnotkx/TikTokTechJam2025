import { Stack } from "expo-router";
import { VideoProvider } from "../context/VideoContext";
import { EditorProvider } from "../context/EditorContext";

export default function RootLayout() {
  return (
    <VideoProvider>
      <EditorProvider>
        <Stack>
          <Stack.Screen name="index" options={{ title: "Home", headerShown: false }} />
          <Stack.Screen name="editor" options={{ title: "Editor", headerShown: true }} />
        </Stack>
      </EditorProvider>
    </VideoProvider>
  );
}
