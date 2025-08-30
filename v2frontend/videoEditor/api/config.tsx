import Constants from "expo-constants";
export const API_URL = (Constants.expoConfig?.extra as any)?.apiUrl as string;