import { Stack } from "expo-router";
import { VideoProvider } from "../context/VideoContext";

export default function RootLayout() {
  return (
    <VideoProvider>
      <Stack>
        <Stack.Screen name="index" options={{ title: "Home", headerShown: false }} />
        <Stack.Screen name="editor" options={{ title: "Editor", headerShown: true }} />
      </Stack>
    </VideoProvider>
  );
}
