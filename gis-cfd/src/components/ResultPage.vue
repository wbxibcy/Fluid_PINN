<template>
    <div>
        <v-snackbar v-model="snackbar" timeout="3000" bottom right color="success">
            {{ snackbarText }}
        </v-snackbar>
        <!-- 结果列表 -->
        <v-container class="results-container" fluid style="padding: 20px 20px;">
            <v-row>
                <v-col cols="12" md="6" v-for="result in results" :key="result.result_id">
                    <v-card>
                        <v-row>
                            <!-- 左侧内容 -->
                            <v-col cols="9">
                                <v-card-title>
                                    {{ result.description }}
                                </v-card-title>
                                <v-card-text>
                                    <p>模拟类型: {{ result.simulation_type }}</p>
                                    <v-btn text color="primary" @click="showGif(result.gif_id)">查看结果</v-btn>
                                </v-card-text>
                            </v-col>

                            <!-- 右侧按钮 -->
                            <v-col cols="3" class="d-flex flex-column align-center justify-center">
                                <v-btn icon color="red" @click="openDeleteDialog(result.result_id)">
                                    <v-icon>mdi-delete</v-icon>
                                </v-btn>
                            </v-col>

                        </v-row>
                    </v-card>
                </v-col>
            </v-row>
        </v-container>

        <v-dialog v-model="deleteDialog" max-width="400">
            <v-card>
                <v-card-title class="text-h6">确认删除</v-card-title>
                <v-card-text>你确定要删除该结果吗？此操作不可撤销。</v-card-text>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn text @click="deleteDialog = false">取消</v-btn>
                    <v-btn color="red" text @click="deleteResult(deleteTargetId)">确定</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>
        <!-- 悬浮的模拟按钮 -->
        <v-btn fab color="blue" dark class="floating-simulate-btn" @click="simulateResult">
            <v-icon>mdi-play-circle</v-icon>
        </v-btn>

        <!-- 模拟表单对话框 -->
        <v-dialog v-model="simulationDialog" max-width="600px">
            <v-card>
                <v-card-title>
                    <span class="text-h5">创建模拟</span>
                </v-card-title>
                <v-card-text>
                    <v-alert v-if="errorMessage" type="error" dense outlined>
                        {{ errorMessage }}
                    </v-alert>

                    <div class="d-flex justify-center my-4" v-if="loading">
                        <v-progress-circular indeterminate color="primary"></v-progress-circular>
                    </div>

                    <v-form ref="simulationForm">
                        <v-textarea v-model="simulationData.description" label="描述" required></v-textarea>
                        <v-select v-model="simulationData.simulation_type" :items="['pinn', 'fvm']" label="模拟类型"
                            required></v-select>

                        <!-- Source 输入 -->
                        <v-text-field v-model="simulationData.source.position" label="源位置 (格式: x1, y1, x2, y2)"
                            required></v-text-field>
                        <v-text-field v-model="simulationData.source.strength" label="源强度" required></v-text-field>

                        <!-- PINN 参数 -->
                        <v-text-field v-if="simulationData.simulation_type === 'pinn'"
                            v-model="simulationData.pinn_params.csv_coordinates" label="CSV 坐标"></v-text-field>
                        <v-file-input v-if="simulationData.simulation_type === 'pinn'" v-model="csvFile"
                            label="上传 CSV 文件" accept=".csv"></v-file-input>

                        <!-- FVM 参数 -->
                        <v-text-field v-if="simulationData.simulation_type === 'fvm'"
                            v-model="simulationData.fvm_params.D" label="扩散系数 D"></v-text-field>
                        <v-text-field v-if="simulationData.simulation_type === 'fvm'"
                            v-model="simulationData.fvm_params.u" label="速度 u"></v-text-field>
                        <v-text-field v-if="simulationData.simulation_type === 'fvm'"
                            v-model="simulationData.fvm_params.v" label="速度 v"></v-text-field>
                        <v-text-field v-if="simulationData.simulation_type === 'fvm'"
                            v-model="simulationData.fvm_params.steps" label="步数"></v-text-field>
                    </v-form>
                </v-card-text>
                <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn color="blue darken-1" text @click="simulationDialog = false">取消</v-btn>
                    <v-btn color="blue darken-1" text @click="createSimulation">开始模拟</v-btn>
                </v-card-actions>
            </v-card>
        </v-dialog>

        <v-dialog v-model="gifDialog" max-width="800px">
            <v-card>
                <v-card-title class="d-flex align-center">
                    <span class="text-h5">室内浓度模拟结果</span>
                    <v-spacer></v-spacer>
                    <v-btn icon @click="gifDialog = false">
                        <v-icon>mdi-close</v-icon>
                    </v-btn>
                </v-card-title>

                <v-card-text class="text-center">
                    <img :src="displayedGifUrl" alt="模拟结果" style="max-width: 100%;" />
                </v-card-text>

            </v-card>
        </v-dialog>
    </div>
</template>

<script>
import { getResultsByFloor, createResult, deleteResultById, getGifById } from '../utils/auth';

export default {
    name: 'ResultPage',
    props: {
        floor_id: {
            type: String,
            required: true,
        },
    },
    data() {
        return {
            deleteDialog: false,
            deleteTargetId: null,
            results: [], // 存储结果列表
            simulationDialog: false, // 控制模拟表单对话框显示
            gifDialog: false, // 控制 GIF 对话框显示
            gifUrl: '', // 存储 GIF 的 URL
            displayedGifUrl: '',
            csvFile: null, // 存储上传的 CSV 文件
            simulationData: {
                floor_id: this.floor_id,
                simulation_type: '',
                description: '',
                source: {
                    position: [],
                    strength: 1.0,
                },
                pinn_params: {
                    csv_coordinates: '',
                },
                fvm_params: {
                    D: 0.01,
                    u: 0.5,
                    v: 0.5,
                    steps: 100,
                },
            },
            loading: false,
            errorMessage: '',
            snackbar: false,
            snackbarText: '',

        };
    },
    created() {
        this.fetchResults();
    },
    watch: {
        gifDialog(val) {
            if (val) {
                this.displayedGifUrl = this.gifUrl + '?t=' + Date.now();
            }
        },
    },
    methods: {
        // 获取结果列表
        fetchResults() {
            const floorId = this.$route.params.floor_id;
            getResultsByFloor(floorId)
                .then((data) => {
                    this.results = data.results;
                })
                .catch((error) => {
                    console.error('获取结果列表失败:', error);
                });
        },

        // 创建模拟
        async createSimulation() {
            this.loading = true;
            this.errorMessage = '';

            try {
                const data = await createResult(this.simulationData, this.csvFile);  // 等待 createResult 完成
                if (data) {
                    this.simulationDialog = false;
                    this.snackbarText = '模拟创建成功！';
                    this.snackbar = true;
                    this.fetchResults();
                } else {
                    this.errorMessage = '模拟创建失败：服务器返回异常';
                }
            } catch (error) {
                console.error('创建模拟失败:', error);
                this.errorMessage = '创建模拟时发生错误：' + (error.message || '未知错误');
            } finally {
                this.loading = false;
            }
        },
        openDeleteDialog(id) {
            this.deleteTargetId = id;
            this.deleteDialog = true;
        },

        // 删除结果
        deleteResult(result_id) {
            deleteResultById(result_id)
                .then(() => {
                    console.log('结果删除成功');
                    this.fetchResults(); // 重新获取结果列表
                    this.deleteDialog = false;
                    this.deleteTargetId = null;
                })
                .catch((error) => {
                    console.error('删除结果失败:', error);
                });
        },

        // 显示 GIF
        showGif(gif_id) {
            getGifById(gif_id)
                .then((data) => {
                    this.gifUrl = data.gif.gif_url;
                    this.gifDialog = true;
                })
                .catch((error) => {
                    console.error('获取 GIF 失败:', error);
                });
        },

        // 打开模拟表单
        simulateResult() {
            this.simulationDialog = true;
        },
    },
};
</script>

<style scoped>
h1 {
    margin-bottom: 20px;
}

.floating-simulate-btn {
    position: fixed;
    bottom: 20px;
    right: 20px;
}

html,
body {
    margin: 0;
    padding: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
}

.results-container {
    background-image: url('@/assets/xxx.jpg');
    background-size: 100% 100%;
    background-position: center;
    background-repeat: no-repeat;
    width: 100vw;
    height: 100vh;
}
</style>