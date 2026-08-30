/// <reference types="vite/client" />

declare module '*.css'
declare module '*.less'

declare interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_AMAP_WEB_JS_KEY: string
  readonly VITE_AMAP_SECURITY_JS_CODE: string
}

declare interface ImportMeta {
  readonly env: ImportMetaEnv
}
