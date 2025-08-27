import '@lynx-js/preact-devtools'
import '@lynx-js/react/debug'
import { root } from '@lynx-js/react'
import { MemoryRouter, Routes, Route } from "react-router";
import { App } from './App.jsx'


root.render(
  <MemoryRouter>
    <Routes>
      <Route path="/" element={<App />} />
    </Routes>
  </MemoryRouter>,
);

if (import.meta.webpackHot) {
  import.meta.webpackHot.accept()
}
