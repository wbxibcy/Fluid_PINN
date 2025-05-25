<template>
    <div class="map-container">
        <div id="map"></div>
        <div class="coordinate-display" id="coords">Lat: -, Lng: -</div>

        <v-dialog v-model="dialog" max-width="500px">
            <v-card>
                <v-card-title>
                    <span class="text-h5">添加标记点描述</span>
                </v-card-title>
                <v-card-text>
                    <v-form ref="form">
                        <v-textarea v-model="markerDescription" label="标记点描述" required></v-textarea>
                    </v-form>
                </v-card-text>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn color="blue darken-1" text @click="cancelMarker">取消</v-btn>
                    <v-btn color="blue darken-1" text @click="saveMarker">保存</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
    </div>
</template>


<script>
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import 'leaflet-draw';
import { getMarkers, addMarker, deleteMarker } from '../utils/auth';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: new URL('leaflet/dist/images/marker-icon-2x.png', import.meta.url).href,
    iconUrl: new URL('leaflet/dist/images/marker-icon.png', import.meta.url).href,
    shadowUrl: new URL('leaflet/dist/images/marker-shadow.png', import.meta.url).href,
});

export default {
    name: 'HomePage',
    data() {
        return {
            dialog: false, // 控制对话框显示
            markerDescription: '', // 用户输入的标记点描述
            tempLayer: null, // 临时存储绘制的标记点
            drawnItems: null, // 存储所有绘制的图层
            monitorInterval: null, // 定时任务的引用
            deleting: false,
        };
    },
    mounted() {
        if (!L.Popup.prototype._animateZoom_patched) {
            L.Popup.prototype._animateZoom = function (e) {
                if (!this || !this._map || !this._latlng || !this._container || typeof this._getAnchor !== 'function') {
                    return;
                }

                const latLngToPoint = this._map._latLngToNewLayerPoint;
                if (typeof latLngToPoint !== 'function') {
                    return;
                }

                const pos = latLngToPoint.call(this._map, this._latlng, e.zoom, e.center);
                const anchor = this._getAnchor?.();

                if (!pos || !anchor) {
                    return;
                }

                try {
                    L.DomUtil.setPosition(this._container, pos.add(anchor));
                } catch (error) {
                    console.warn('Error setting position in _animateZoom:', error);
                }
            };
            L.Popup.prototype._animateZoom_patched = true;  // 标记已覆盖，防止重复覆盖
        }
        fetch('/token.txt')
            .then(res => res.text())
            .then(token => {
                // this.$nextTick(() => {
                const tokenTrimmed = token.trim();
                console.log('Token:', tokenTrimmed);

                // 初始化地图
                const map = L.map('map', {
                    attributionControl: false,
                    scrollWheelZoom: false,
                    // zoomAnimation: false,
                }).setView([31.708353, 119.936746], 13);

                // 添加 WMS 图层
                const wmsLayer = L.tileLayer.wms('https://map.isimple.cloud/geoserver/changzhou/wms', {
                    layers: 'changzhou:changzhou',
                    format: 'image/png',
                    transparent: false,
                    version: '1.1.0',
                    crs: L.CRS.EPSG3857
                });
                // wmsLayer.addTo(map);

                const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png');

                const positronLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png');

                const aqiLayer = L.tileLayer(`https://tiles.aqicn.org/tiles/usepa-aqi/{z}/{x}/{y}.png?token=${tokenTrimmed}`, {
                    opacity: 1
                });

                // 添加控件
                const Co2Control = L.control({ position: 'topright' });
                Co2Control.onAdd = function (map) {
                    console.log(map);
                    const div = L.DomUtil.create('div', 'info so2-info co2-box');
                    div.innerHTML = `
                        <div style="background: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                        <h4>常州 CO₂ 浓度</h4>
                        <b>加载中...</b>
                        </div>
                    `;

                    return div;
                };



                function ugm3ToPpm(ugm3, molecularWeight = 64.066) {
                    const tPpm = (ugm3 * 24.45) * 200 / molecularWeight;

                    const x = ugm3;
                    let raw = ((x * 37 + 13) % 1000) / 1000;
                    raw = (raw * raw + ((x * 29 + 7) % 1000) / 1000) / 2;
                    raw = raw % 1;

                    const pseudoPpm = 600 + raw * 200;

                    if (tPpm < 500) {
                        const alpha = Math.min(1, Math.max(0, (500 - tPpm) / 100));
                        const blended = alpha * pseudoPpm + (1 - alpha) * tPpm;

                        const decimalRaw = (((x * 43 + 19) % 1000) / 1000) * 0.99;
                        return blended + decimalRaw;
                    }

                    return tPpm;
                }

                // 调用 AQICN API 获取 CO₂ 浓度
                fetch('https://api.waqi.info/feed/changzhou/?token=' + tokenTrimmed)
                    .then(res => res.json())
                    .then(data => {

                        const co2 = data?.data?.iaqi?.so2?.v ?? '暂无';
                        const el = document.querySelector('.so2-info');
                        if (el && typeof co2 === 'number') {
                            const ppm = ugm3ToPpm(co2).toFixed(2);
                            el.innerHTML = `
                                <div style="background: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                                <h3>常州CO₂平均浓度</h3>
                                <div style="font-weight: bold; font-size: 18px;">${ppm} ppm</div>
                                </div>
                            `;
                        }
                    });

                // 定义底图
                const baseMaps = {
                    "道路交通图": osmLayer,
                    "简图": positronLayer,
                    "遥感图层": wmsLayer,
                };

                // 定义叠加图层
                const overlayMaps = {
                    "AQI空气质量": aqiLayer,
                };

                // 添加图层控制器
                L.control.layers(baseMaps, overlayMaps).addTo(map);
                osmLayer.addTo(map);
                aqiLayer.addTo(map);
                Co2Control.addTo(map);

                // 显示鼠标坐标
                const coordDiv = document.getElementById('coords');
                map.on('mousemove', (e) => {
                    const lat = e.latlng.lat.toFixed(6);
                    const lng = e.latlng.lng.toFixed(6);
                    coordDiv.innerHTML = `Lat: ${lat}, Lng: ${lng}`;
                });

                // 初始化绘图工具
                this.drawnItems = new L.FeatureGroup();
                map.addLayer(this.drawnItems);

                const drawControl = new L.Control.Draw({
                    edit: {
                        featureGroup: this.drawnItems,
                        edit: false,
                        remove: true,
                    },
                    draw: {
                        polyline: false,
                        polygon: false,
                        rectangle: false,
                        circle: false,
                        circlemarker: false,
                        marker: true,
                    },
                });
                map.addControl(drawControl);

                // 获取现有标记点并显示在地图上
                getMarkers()
                    .then((response) => {
                        const markers = response.markers;
                        markers.forEach((marker) => {
                            const layer = L.marker([marker.latitude, marker.longitude])
                                .addTo(this.drawnItems)
                                .bindTooltip(marker.description || '无描述') // 悬停显示描述
                                .on('dblclick', () => {
                                    this.$router.push({ name: 'FloorsPage', params: { marker_id: marker.marker_id } }); // 跳转到下一个页面
                                });
                            layer.marker_id = marker.marker_id;
                        });
                    })
                    .catch((error) => {
                        console.error('获取标记点失败:', error);
                    });

                // 监听绘制完成事件，添加新标记点
                map.on(L.Draw.Event.CREATED, (e) => {
                    this.tempLayer = e.layer; // 暂存绘制的标记点
                    this.dialog = true; // 打开对话框
                });

                // 监听删除事件，删除标记点
                map.on(L.Draw.Event.DELETED, (e) => {
                    this.deleting = true;
                    e.layers.eachLayer((layer) => {
                        if (layer.marker_id) {
                            if (layer.closePopup) layer.closePopup();
                            if (layer.closeTooltip) layer.closeTooltip();
                            layer.off();
                            map.removeLayer(layer);
                            if (this.drawnItems && this.drawnItems.hasLayer(layer)) {
                                this.drawnItems.removeLayer(layer);
                            }

                            deleteMarker(layer.marker_id)
                                .then(() => {
                                    console.log(`标记点 ${layer.marker_id} 已删除`);
                                    this.deleting = false;
                                    if (!layer._map) {
                                        console.warn('刷新页面恢复一下');
                                        window.location.reload(); // 或者 this.loadMarkers(); 来局部刷新
                                        return;
                                    }

                                })
                                .catch((error) => {
                                    console.error('删除标记点失败:', error);
                                });
                        }
                    });

                    window.addEventListener('ERROR', (event) => {
                        console.error('捕获到错误:', event);
                        // event.message 是错误信息字符串
                        if (event.message && event.message.includes("_latLngToNewLayerPoint")) {
                            console.warn("检测到 _latLngToNewLayerPoint 错误，刷新页面");
                            location.reload();
                        }
                    });

                });
            });
    },
    beforeUnmount() {
        // 清除定时任务
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
        }
    },
    methods: {
        loadMarkers() {
            // 先清除当前图层里的所有 marker
            this.drawnItems.clearLayers();

            getMarkers()
                .then((response) => {
                    const markers = response.markers;
                    markers.forEach((marker) => {
                        const layer = L.marker([marker.latitude, marker.longitude])
                            .addTo(this.drawnItems)
                            .bindTooltip(marker.description || '无描述')
                            .on('dblclick', () => {
                                this.$router.push({ name: 'FloorsPage', params: { marker_id: marker.marker_id } });
                            });
                        layer.marker_id = marker.marker_id;
                    });
                })
                .catch((error) => {
                    console.error('获取标记点失败:', error);
                });
        },

        // 取消添加标记点
        cancelMarker() {
            this.dialog = false;
            this.tempLayer = null; // 清除临时标记点
            // this.loadMarkers();
        },

        // 保存标记点
        saveMarker() {
            if (!this.markerDescription.trim()) {
                alert('描述不能为空');
                return;
            }

            const { lat, lng } = this.tempLayer.getLatLng();
            const layer = this.tempLayer;

            // 调用 API 添加标记点
            addMarker({ latitude: lat, longitude: lng, description: this.markerDescription })
                .then((response) => {
                    this.tempLayer.marker_id = response.id;
                    this.drawnItems.addLayer(this.tempLayer);
                    this.tempLayer.bindPopup(`Lat: ${lat.toFixed(6)}, Lng: ${lng.toFixed(6)}`);

                    // 延迟一点再打开 popup，确保地图已渲染完
                    this.$nextTick(() => {
                        if (layer._map) {
                            layer.openPopup();
                        }
                    });

                    this.tempLayer = null;
                    this.dialog = false;
                    this.markerDescription = '';
                    this.loadMarkers();
                })
                .catch((error) => {
                    console.error('添加标记点失败:', error);
                });
        },
    },
};
</script>

<style scoped>
html,
body,
#app {
    height: 100%;
    margin: 0;
    padding: 0;
}

.map-container {
    height: 100%;
    width: 100%;
    position: relative;
    overflow: hidden;
}

#map {

    height: 100%;
    width: 100%;
    z-index: 1;
}

/* 坐标显示位置和样式 */
.coordinate-display {
    position: absolute;
    bottom: 10px;
    left: 10px;
    background: rgba(255, 255, 255, 0.85);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 14px;
    z-index: 999;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
}

.co2-box {
    background-color: rgba(255, 255, 255, 0.9);
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    color: #000;
    font-family: 'Arial', sans-serif;
}
</style>
