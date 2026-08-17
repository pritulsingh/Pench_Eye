import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import Triage from './pages/Triage'
import Upload from './pages/Upload'
import Tigers from './pages/Tigers'
import TigerProfile from './pages/TigerProfile'
import Observations from './pages/Observations'
import Reviews from './pages/Reviews'
import Cameras from './pages/Cameras'
import CameraDetail from './pages/CameraDetail'
import MapView from './pages/MapView'
import Alerts from './pages/Alerts'
import Analytics from './pages/Analytics'
import Gallery from './pages/Gallery'
import NotFound from './pages/NotFound'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <Layout title="Command Center" subtitle="Pench Eye — wildlife intelligence overview">
              <Dashboard />
            </Layout>
          }
        />
        <Route
          path="/map"
          element={
            <Layout title="Reserve Map" subtitle="Zones, camera network, sightings and movement">
              <MapView />
            </Layout>
          }
        />
        <Route
          path="/cameras"
          element={
            <Layout title="Camera Traps" subtitle="Monitoring network status and activity">
              <Cameras />
            </Layout>
          }
        />
        <Route
          path="/cameras/:cameraId"
          element={
            <Layout title="Camera Station" subtitle="Location, activity and recent captures">
              <CameraDetail />
            </Layout>
          }
        />
        <Route
          path="/tigers"
          element={
            <Layout title="Tiger Catalog" subtitle="Individual identification database">
              <Tigers />
            </Layout>
          }
        />
        <Route
          path="/tigers/:id"
          element={
            <Layout title="Tiger Profile" subtitle="Sightings, cameras and movement history">
              <TigerProfile />
            </Layout>
          }
        />
        <Route
          path="/observations"
          element={
            <Layout title="Detections" subtitle="Every recorded sighting with filters">
              <Observations />
            </Layout>
          }
        />
        <Route
          path="/gallery"
          element={
            <Layout title="Image Gallery" subtitle="Browse camera-trap captures and metadata">
              <Gallery />
            </Layout>
          }
        />
        <Route
          path="/alerts"
          element={
            <Layout title="Alerts" subtitle="Rule-based monitoring signals">
              <Alerts />
            </Layout>
          }
        />
        <Route
          path="/analytics"
          element={
            <Layout title="Analytics" subtitle="Detection patterns across the reserve">
              <Analytics />
            </Layout>
          }
        />
        <Route
          path="/triage"
          element={
            <Layout title="Image Triage" subtitle="Blank-frame filtering and quarantine">
              <Triage />
            </Layout>
          }
        />
        <Route
          path="/upload"
          element={
            <Layout title="Ingest Images" subtitle="Run the camera-trap pipeline on new frames">
              <Upload />
            </Layout>
          }
        />
        <Route
          path="/reviews"
          element={
            <Layout title="Human Review" subtitle="Ambiguous identity decisions">
              <Reviews />
            </Layout>
          }
        />
        <Route
          path="*"
          element={
            <Layout title="Page not found">
              <NotFound />
            </Layout>
          }
        />
      </Routes>
    </BrowserRouter>
  )
}
