import { Text, View } from "react-native";
import { useVideo } from "../context/VideoContext";
import { Video } from "expo-av";
import * as FileSystem from 'expo-file-system';

export default function Editor() {
    const {videoUri, setVideoUri} = useVideo();

    const getVideoData = async (videoUri: string) => {
        const base64 = await FileSystem.readAsStringAsync(videoUri, {
            encoding: FileSystem.EncodingType.Base64
        });

        // If using ffmpeg.wasm
        const uint8 = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
        return uint8;
    };

    return (

        <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
            <Text>Edit your video</Text> 
            {videoUri && (
            <Video
                source={{ uri: videoUri }}
                style={{ width: "90%", height: 200 }}
                useNativeControls
            />
            )}
        </View> 
    );
}