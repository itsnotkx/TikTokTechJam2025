import { Text, View } from "react-native";
import { useVideo } from "../context/VideoContext";
import { Video } from "expo-av";

export default function Editor() {
    const {videoUri, setVideoUri} = useVideo();

    return <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        
        <Text>Edit your video</Text>
        {videoUri && (
        <Video
          source={{ uri: videoUri }}
          style={{ width: "100%", height: 300 }}
          useNativeControls
        />
      )}
        
    </View>
}