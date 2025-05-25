<template>
    <div>
        <v-container class="floors-container" fluid style="padding: 20px 20px;">
            <v-row>
                <v-col cols="12" md="4" v-for="floor in floors" :key="floor.id">
                    <v-card @click="showGeojson(floor.geojson_id)">
                        <v-card-title>
                            {{ floor.name }}
                            <v-spacer></v-spacer>
                        </v-card-title>
                        <v-card-text>{{ floor.description }}</v-card-text>
                        <v-card-actions style="justify-content: flex-end;">
                            <v-btn icon color="green" variant="outlined" style="background-color: transparent;"
                                @click.stop="simulateFloor(floor.floor_id)">
                                <v-icon>mdi-tailwind</v-icon>
                            </v-btn>

                            <!-- 点击时打开确认删除弹窗 -->
                            <v-btn icon color="red" variant="outlined" style="background-color: transparent;"
                                @click.stop="openDeleteDialog(floor.floor_id, floor.geojson_id)">
                                <v-icon>mdi-delete</v-icon>
                            </v-btn>
                        </v-card-actions>
                    </v-card>
                </v-col>
            </v-row>

            <!-- 删除确认弹窗 -->
            <v-dialog v-model="deleteDialog" max-width="400">
                <v-card>
                    <v-card-title class="headline">确认删除</v-card-title>
                    <v-card-text>确定要删除该楼层吗？此操作不可恢复。</v-card-text>
                    <v-card-actions>
                        <v-spacer></v-spacer>
                        <v-btn text @click="deleteDialog = false">取消</v-btn>
                        <v-btn color="red" text
                            @click.stop="deleteFloor(deleteTarget.floorId, deleteTarget.geojsonId)">删除</v-btn>
                    </v-card-actions>
                </v-card>
            </v-dialog>
        </v-container>


        <!-- GeoJSON 展示对话框 -->
        <v-dialog v-model="geojsonDialog" max-width="600px">
            <v-card>
                <v-card-title style="display: flex; align-items: center;">
                    <span class="text-h5">GeoJSON 数据展示</span>
                    <v-spacer></v-spacer>
                    <v-btn icon @click="geojsonDialog = false">
                        <v-icon>mdi-close</v-icon>
                    </v-btn>
                </v-card-title>

                <v-card-text class="d-flex justify-center">
                    <canvas id="geojsonCanvas" width="400" height="400"
                        style="border: 1px solid black; background-color: #f9f9f9;"></canvas>
                </v-card-text>

            </v-card>
        </v-dialog>

        <!-- 新增户型对话框 -->
        <v-dialog v-model="dialog" max-width="800px">
            <v-card>
                <v-card-title>
                    <span class="text-h5">新增户型</span>
                </v-card-title>
                <v-card-text>
                    <v-form ref="form">
                        <v-text-field v-model="form.name" label="户型名称" />
                        <v-text-field v-model="form.description" label="户型描述" />
                        <v-text-field v-model="form.width" label="边界宽度 (Width)(最大值为15)" />
                        <v-text-field v-model="form.height" label="边界长度 (Height)(最大值为15)" />
                        <v-text-field v-model="form.inletStart" label="入口起点 格式: x,y" />
                        <v-text-field v-model="form.inletEnd" label="入口终点 格式: x,y" />
                        <v-text-field v-model="form.outletStart" label="出口起点 格式: x,y" />
                        <v-text-field v-model="form.outletEnd" label="出口终点 格式: x,y" />

                    </v-form>
                </v-card-text>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn color="blue darken-1" text @click="dialog = false">取消</v-btn>
                    <v-btn color="blue darken-1" text @click="confirmCoordinates">确认添加</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>

        <!-- 添加按钮 -->
        <v-btn fab color="primary" dark class="floating-btn" @click="dialog = true">
            <v-icon>mdi-plus</v-icon>
        </v-btn>
    </div>
</template>

<script>
import { getFloors, deleteFloor, addGeojson, addFloor, deleteGeojson, getGeojson } from '../utils/auth';

export default {
    name: 'FloorsPage',
    data() {
        return {
            floors: [],
            marker_id: null,
            dialog: false, // 控制新增户型对话框显示
            geojsonDialog: false, // 控制 GeoJSON 弹窗显示
            // form: {
            //     name: '',
            //     description: '',
            //     boundary: '', // 用户输入的边界坐标
            //     inlet: '', // 用户输入的入口坐标
            //     outlet: '', // 用户输入的出口坐标
            // },
            form: {
                name: '',
                description: '',
                width: '',       // 宽
                height: '',      // 长
                inletStart: '',  // [x, y]
                inletEnd: '',
                outletStart: '',
                outletEnd: ''
            },
            deleteDialog: false,
            deleteTarget: {
                floorId: null,
                geojsonId: null,
            },
        };
    },
    created() {
        this.marker_id = this.$route.params.marker_id; // 获取传递的 marker_id
        this.fetchFloors();
    },
    watch: {
        // 监听 floors 数据的变化
        floors: {
            handler(newVal, oldVal) {
                console.log('floors 数据发生变化:');
                console.log('旧值:', oldVal);
                console.log('新值:', newVal);
            },
            deep: true,
        },
    },
    methods: {
        openDeleteDialog(floorId, geojsonId) {
            this.deleteTarget.floorId = floorId;
            this.deleteTarget.geojsonId = geojsonId;
            this.deleteDialog = true;
        },
        simulateFloor(floor_id) {
            if (!floor_id) {
                console.error('floor_id 不存在，无法跳转到模拟页面');
                return;
            }

            // 跳转到 ResultPage，并传递 floor_id
            this.$router.push({ name: 'ResultPage', params: { floor_id } });
        },

        // 获取户型数据
        fetchFloors() {
            getFloors(this.marker_id)
                .then((response) => {
                    this.floors = response.floors; // 假设返回数据包含 floors 数组
                })
                .catch((error) => {
                    console.error('获取户型数据失败:', error);
                });
        },
        // 删除户型
        deleteFloor(floor_id, geojson_id) {
            if (geojson_id) {
                // 先删除 GeoJSON 数据
                deleteGeojson(geojson_id)
                    .then(() => {
                        console.log(`GeoJSON 数据 ${geojson_id} 已删除`);
                        // 再删除 Floor 数据
                        return deleteFloor(floor_id);
                    })
                    .then(() => {
                        console.log(`户型数据 ${floor_id} 已删除`);
                        // 更新本地 floors 列表
                        this.floors = this.floors.filter((floor) => floor.floor_id !== floor_id);

                        this.deleteDialog = false;
                    })
                    .catch((error) => {
                        console.error('删除失败:', error);
                    });
            } else {
                // 如果没有 geojson_id，直接删除 Floor 数据
                deleteFloor(floor_id)
                    .then(() => {
                        console.log(`户型数据 ${floor_id} 已删除`);
                        this.floors = this.floors.filter((floor) => floor.floor_id !== floor_id);
                    })
                    .catch((error) => {
                        console.error('删除户型失败:', error);
                    });
            }
        },
        // 确认添加前显示坐标
        confirmCoordinates() {
            const { width, height, inletStart, inletEnd, outletStart, outletEnd } = this.form;

            if (!width || !height || !inletStart || !inletEnd || !outletStart || !outletEnd) {
                alert('请填写完整的边界、入口和出口坐标！');
                return;
            }

            // console.log('确认的边界坐标:', this.form.boundary);
            // console.log('确认的入口坐标:', this.form.inlet);
            // console.log('确认的出口坐标:', this.form.outlet);

            // 提交表单
            this.submitForm();
        },

        // 提交表单并新增户型
        submitForm() {
            if (!this.$refs.form.validate()) {
                return;
            }

            try {
                // 将用户输入的坐标解析为 JSON 格式
                const width = parseFloat(this.form.width);
                const height = parseFloat(this.form.height);
                if (isNaN(width) || isNaN(height)) {
                    alert('请输入有效的边界长度和宽度');
                    return;
                }
                const boundary = [[[0, 0], [0, height], [width, height], [width, 0], [0, 0]]];
                // const boundary = JSON.parse(this.form.boundary);

                const inletStart = this.form.inletStart.split(',').map(Number);
                const inletEnd = this.form.inletEnd.split(',').map(Number);
                const outletStart = this.form.outletStart.split(',').map(Number);
                const outletEnd = this.form.outletEnd.split(',').map(Number);

                const inlet = [inletStart, inletEnd];
                const outlet = [outletStart, outletEnd];

                // const inlet = JSON.parse(this.form.inlet);
                // const outlet = JSON.parse(this.form.outlet);

                const isOnBoundary = (point) => {
                    const [x, y] = point;
                    return (
                        (x === 0 || x === width) && y >= 0 && y <= height ||  // 左右边
                        (y === 0 || y === height) && x >= 0 && x <= width     // 上下边
                    );
                };

                if (![...inlet, ...outlet].every(isOnBoundary)) {
                    alert('入口或出口必须在边界上（矩形的边缘）');
                    return;
                }

                // 构造 GeoJSON 数据
                const geojsonData = {
                    type: 'FeatureCollection',
                    features: [
                        {
                            type: 'Feature',
                            properties: { type: 'boundary' },
                            geometry: {
                                type: 'Polygon',
                                coordinates: boundary,
                            },
                        },
                        {
                            type: 'Feature',
                            properties: { type: 'inlet' },
                            geometry: {
                                type: 'LineString',
                                coordinates: inlet,
                            },
                        },
                        {
                            type: 'Feature',
                            properties: { type: 'outlet' },
                            geometry: {
                                type: 'LineString',
                                coordinates: outlet,
                            },
                        },
                    ],
                };

                // 先新增 GeoJSON
                addGeojson({ geojson_data: geojsonData })
                    .then((geojsonResponse) => {
                        const geojson_id = geojsonResponse.geojson.geojson_id;

                        // 构造 Floor 数据
                        const floorData = {
                            marker_id: this.marker_id,
                            geojson_id: geojson_id,
                            name: this.form.name,
                            description: this.form.description,
                        };

                        // 再新增 Floor
                        return addFloor(floorData);
                    })
                    .then((floorResponse) => {
                        // 将新户型添加到列表中
                        this.floors.push(floorResponse);
                        this.fetchFloors(); // 重新获取户型数据
                        this.dialog = false; // 关闭对话框
                        this.resetForm(); // 重置表单
                    })
                    .catch((error) => {
                        console.error('新增户型失败:', error);
                    });
            } catch (error) {
                console.error('表单数据解析失败:', error);
                alert('输入的坐标格式有误，请检查后重试');
            }
        },

        // 重置表单
        resetForm() {
            this.form.name = '';
            this.form.description = '';
            this.form.width = '';
            this.form.height = '';
            this.form.inletStart = '';
            this.form.inletEnd = '';
            this.form.outletStart = '';
            this.form.outletEnd = '';
        },
        // 获取并展示 GeoJSON 数据
        showGeojson(geojson_id) {
            if (!geojson_id) {
                alert('该户型没有关联的 GeoJSON 数据');
                return;
            }

            // 调用 API 获取 GeoJSON 数据
            getGeojson(geojson_id)
                .then((response) => {
                    const geojsonData = response.geojson.geojson_data;
                    this.geojsonDialog = true; // 打开弹窗
                    this.$nextTick(() => {
                        this.drawGeojson(geojsonData); // 调用绘制方法
                    });
                })
                .catch((error) => {
                    console.error('获取 GeoJSON 数据失败:', error);
                    alert('获取 GeoJSON 数据失败，请稍后重试');
                });
        },

        drawGeojson(geojsonData) {
            const canvas = document.getElementById('geojsonCanvas');
            const ctx = canvas.getContext('2d');

            // 清空画布
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 缩放比例
            const scale = 20;

            // 1. 计算所有点的边界
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

            geojsonData.features.forEach((feature) => {
                const coords = feature.geometry.coordinates.flat(Infinity);
                for (let i = 0; i < coords.length; i += 2) {
                    const x = coords[i], y = coords[i + 1];
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            });

            // 2. 计算中心点和偏移量
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const offsetX = canvas.width / 2;
            const offsetY = canvas.height / 2;

            // 3. 绘图
            geojsonData.features.forEach((feature) => {
                const { type } = feature.geometry;
                const coordinates = feature.geometry.coordinates;
                const featureType = feature.properties.type;

                // 设置样式
                if (featureType === 'boundary') {
                    ctx.strokeStyle = '#BEDAFF';
                    ctx.fillStyle = '#E8F3FF';
                } else if (featureType === 'inlet') {
                    ctx.strokeStyle = '#CB1E83';
                } else if (featureType === 'outlet') {
                    ctx.strokeStyle = '#551DB0';
                }
                ctx.lineWidth = 3;

                if (type === 'Polygon') {
                    coordinates.forEach((ring) => {
                        drawPolygon(ring, ctx);
                    });
                } else if (type === 'LineString') {
                    ctx.beginPath();
                    coordinates.forEach(([x, y], index) => {
                        const sx = (x - centerX) * scale + offsetX;
                        const sy = offsetY - (y - centerY) * scale; // y 轴向上
                        if (index === 0) {
                            ctx.moveTo(sx, sy);
                        } else {
                            ctx.lineTo(sx, sy);
                        }
                    });
                    ctx.stroke();
                }
            });

            // 绘制多边形函数
            function drawPolygon(ring, context) {
                context.beginPath();
                ring.forEach(([x, y], index) => {
                    const sx = (x - centerX) * scale + offsetX;
                    const sy = offsetY - (y - centerY) * scale;
                    if (index === 0) {
                        context.moveTo(sx, sy);
                    } else {
                        context.lineTo(sx, sy);
                    }
                });
                context.closePath();
                context.fill();
                context.stroke();
            }
        },


        // 设置绘制模式
        setMode(mode) {
            this.mode = mode;
            console.log(`当前模式: ${mode}`);
        },

        // 开始绘制
        startDrawing(e) {
            this.isDrawing = true;
            const { x, y } = this.getMousePosition(e);
            this.currentShape.push([x, y]); // 记录起点
        },

        // 绘制过程
        draw(e) {
            if (!this.isDrawing) return;

            const { x, y } = this.getMousePosition(e);
            const lastPoint = this.currentShape[this.currentShape.length - 1];

            // 绘制线段
            this.ctx.beginPath();
            this.ctx.moveTo(lastPoint[0], lastPoint[1]);
            this.ctx.lineTo(x, y);

            // 根据模式设置颜色
            if (this.mode === 'boundary') {
                this.ctx.strokeStyle = '#BEDAFF';
            } else if (this.mode === 'inlet') {
                this.ctx.strokeStyle = '#CB1E83';
            } else if (this.mode === 'outlet') {
                this.ctx.strokeStyle = '#551DB0';
            }

            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            this.ctx.closePath();

            // 记录当前点
            this.currentShape.push([x, y]);
        },

        // 停止绘制
        stopDrawing() {
            if (!this.isDrawing) return;

            this.isDrawing = false;

            // 保存当前图形到对应的数组
            if (this.mode === 'boundary') {
                this.boundary.push([...this.currentShape]);
            } else if (this.mode === 'inlet') {
                this.inlet.push([...this.currentShape]);
            } else if (this.mode === 'outlet') {
                this.outlet.push([...this.currentShape]);
            }

            this.currentShape = []; // 清空当前图形
        },

        // 获取鼠标在 Canvas 上的坐标
        getMousePosition(e) {
            const rect = this.canvas.getBoundingClientRect();
            return {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
            };
        },

        // 获取坐标数据
        submitData() {
            console.log('边界坐标:', this.boundary);
            console.log('入口坐标:', this.inlet);
            console.log('出口坐标:', this.outlet);

            const geojsonData = {
                type: 'FeatureCollection',
                features: [
                    ...this.boundary.map((shape) => ({
                        type: 'Feature',
                        properties: { type: 'boundary' },
                        geometry: {
                            type: 'Polygon',
                            coordinates: [shape.map(([x, y]) => [x / 10, y / 10])], // 缩放坐标
                        },
                    })),
                    ...this.inlet.map((shape) => ({
                        type: 'Feature',
                        properties: { type: 'inlet' },
                        geometry: {
                            type: 'LineString',
                            coordinates: shape.map(([x, y]) => [x / 10, y / 10]), // 缩放坐标
                        },
                    })),
                    ...this.outlet.map((shape) => ({
                        type: 'Feature',
                        properties: { type: 'outlet' },
                        geometry: {
                            type: 'LineString',
                            coordinates: shape.map(([x, y]) => [x / 10, y / 10]), // 缩放坐标
                        },
                    })),
                ],
            };

            console.log('生成的 GeoJSON 数据:', geojsonData);
        },

        // 清空画布
        clearCanvas() {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.boundary = [];
            this.inlet = [];
            this.outlet = [];
        },
    },
};
</script>

<style scoped>
html,
body {
    margin: 0;
    padding: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
}

.floors-container {
    background-image: url('@/assets/xxx.jpg');
    background-size: 100% 100%;
    background-position: center;
    background-repeat: no-repeat;
    width: 100vw;
    height: 100vh;
}

.floating-btn {
    position: fixed;
    bottom: 16px;
    right: 16px;
}

#geojsonCanvas {
    margin-top: 20px;
    display: block;
    background-color: transparent;
}
</style>