import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { CommonCodeProvider } from '@/contexts/CommonCodeContext.jsx'

createRoot(document.getElementById('root')).render(
  <CommonCodeProvider>
    <App />
  </CommonCodeProvider>
)
