import React, { useState } from "react";
import { Button, View, ActivityIndicator, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { router } from "expo-router";
import { useVideo } from "../context/VideoContext";
import { api } from "../api/video";

export default function Index() {
  const { upload, setVideoUri, setUpload } = useVideo();
  const [busy, setBusy] = useState(false);

  const startVideoJob = async (videoUri: string) => {
    const form = new FormData();
    form.append("file", {
      uri: videoUri,
      name: "clip.mp4",
      type: "video/mp4",
    } as any);
    const res = await api.uploadVideo(form); // <-- uses FormData-safe endpoint
    return res;                                   // <-- return a string, not a cast
  };

  const pickVideo = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: true,
      quality: 1,
    });
    if (result.canceled) return;

    const uri = result.assets[0].uri;
    setBusy(true);
    try {
      setVideoUri(uri);
      setUpload({ status: "uploading" });

      const {jobId, videoUri, pill_objects} = await startVideoJob(uri);       // <-- now a string
      setUpload({ status: "processing", jobId });       // <-- use a valid status from your union

      console.log("Upload started:", jobId);
      console.log("Pill objects:", pill_objects);
      console.log("Upload status:", upload.status);

      router.push("/editor");
    } catch (e: any) {
      const message = e?.message ?? "Unknown error";
      setUpload({ status: "error", message });
      Alert.alert("Upload failed", message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={{ justifyContent: "center", alignItems: "center", flex: 1 }}>
      <Button title={busy ? "Starting…" : "Upload Video"} onPress={pickVideo} disabled={busy} />
      {busy && <ActivityIndicator style={{ marginTop: 12 }} />}
    </View>
  );
}
