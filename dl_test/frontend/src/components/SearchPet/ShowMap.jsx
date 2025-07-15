import React, { useEffect, useState } from 'react';

const KAKAO_MAP_API_KEY = '9a19ecb7e00a6db6be9d34fd5f9f3c09';

const KakaoMap = ({ selectedLocation, nearestShelter }) => {
  const [map, setMap] = useState(null);
  const [marker, setMarker] = useState(null);
  const [coords, setCoords] = useState({ lat: 37.26644, lng: 127.000609 }); // 기본 수원역
  const [address, setAddress] = useState('');
  const [polyline, setPolyline] = useState(null);

  useEffect(() => {
    const existingScript = document.getElementById('kakao-map-script');
    if (existingScript) existingScript.remove();

    const script = document.createElement('script');
    script.id = 'kakao-map-script';
    script.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${KAKAO_MAP_API_KEY}&libraries=services&autoload=false`; 
    script.async = true;
    script.onload = initMap;
    document.head.appendChild(script);

    function initMap() {
      window.kakao.maps.load(() => {
        const container = document.getElementById('map');
        const options = {
          center: new window.kakao.maps.LatLng(coords.lat, coords.lng),
          level: 7,
        };
        const mapInstance = new window.kakao.maps.Map(container, options);
        setMap(mapInstance);

        const markerInstance = new window.kakao.maps.Marker({
          position: options.center,
          map: mapInstance,
          draggable: true,
        });
        setMarker(markerInstance);

        const geocoder = new window.kakao.maps.services.Geocoder();

        const searchAddrFromCoords = (lat, lng) => {
          const coord = new window.kakao.maps.LatLng(lat, lng);
          geocoder.coord2Address(coord.getLng(), coord.getLat(), (result, status) => {
            if (status === window.kakao.maps.services.Status.OK) {
              const roadAddr = result[0].road_address?.address_name;
              setAddress(roadAddr || '');
            } else {
              setAddress('');
            }
          });
        };

        searchAddrFromCoords(coords.lat, coords.lng);

        window.kakao.maps.event.addListener(markerInstance, 'dragend', function () {
          const pos = markerInstance.getPosition();
          const newCoords = { lat: pos.getLat(), lng: pos.getLng() };
          setCoords(newCoords);
          searchAddrFromCoords(newCoords.lat, newCoords.lng);
        });

        window.kakao.maps.event.addListener(mapInstance, 'click', function (mouseEvent) {
          const latlng = mouseEvent.latLng;
          markerInstance.setPosition(latlng);
          const newCoords = { lat: latlng.getLat(), lng: latlng.getLng() };
          setCoords(newCoords);
          searchAddrFromCoords(newCoords.lat, newCoords.lng);
        });
      });
    }
  }, []);

  // 선택된 위치 반영
  useEffect(() => {
    if (selectedLocation && marker && map) {
      const moveLatLon = new window.kakao.maps.LatLng(selectedLocation.lat, selectedLocation.lng);
      marker.setPosition(moveLatLon);
      map.setCenter(moveLatLon);
      setCoords(selectedLocation);

      const geocoder = new window.kakao.maps.services.Geocoder();
      geocoder.coord2Address(
        selectedLocation.lng,
        selectedLocation.lat,
        (result, status) => {
          if (status === window.kakao.maps.services.Status.OK) {
            const roadAddr = result[0].road_address?.address_name;
            setAddress(roadAddr || '');
          } else {
            setAddress('');
          }
        }
      );
    }
  }, [selectedLocation, marker, map]);

  // 보호소 마커 추가
  useEffect(() => {
    if (map && nearestShelter) {
      const shelterLatLng = new window.kakao.maps.LatLng(nearestShelter.latitude, nearestShelter.longitude);

      const shelterMarker = new window.kakao.maps.Marker({
        position: shelterLatLng,
        map: map,
        image: new window.kakao.maps.MarkerImage(
          'https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/marker_red.png',
          new window.kakao.maps.Size(24, 35),
          { offset: new window.kakao.maps.Point(12, 35) }
        ),
      });

      // infoWindow.open(map, shelterMarker);

      return () => {
        shelterMarker.setMap(null);
        // infoWindow.close();
      };
    }
  }, [nearestShelter, map]);

  useEffect(() => {
    if (map && selectedLocation && nearestShelter) {
      if (polyline) {
        polyline.setMap(null);
      }
      const linePath = [
        new window.kakao.maps.LatLng(selectedLocation.lat, selectedLocation.lng),
        new window.kakao.maps.LatLng(nearestShelter.latitude, nearestShelter.longitude),
      ];
      const newPolyline = new window.kakao.maps.Polyline({
        path: linePath,
        strokeWeight: 5,
        strokeColor: '#FF0000',
        strokeOpacity: 0.7,
        strokeStyle: 'solid',
      });
      newPolyline.setMap(map);
      setPolyline(newPolyline);
    }
  }, [map, selectedLocation, nearestShelter]);

  return (
    <div style={{ padding: '20px' }}>
      <p>
        * 마커를 드래그하거나 지도를 클릭하여 위치를 이동할 수 있습니다.
        <br />
        현재 좌표: 위도 {coords.lat.toFixed(6)}, 경도 {coords.lng.toFixed(6)}
      </p>
      {address && <p>도로명 주소: {address}</p>}
      <div id="map" style={{ width: '100%', height: '475px', marginTop: '16px' }}></div>
    </div>
  );
};

export default KakaoMap;
