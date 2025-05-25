import axios from 'axios'
import router from '../router'

const API = axios.create({
    baseURL: 'http://localhost:8000',
})

// 从本地存储中获取 token
function getToken() {
    return localStorage.getItem('token') || null
}

// 将 token 存储到本地存储
function setToken(token) {
    localStorage.setItem('token', token)
}

// 清除本地存储中的 token
function clearToken() {
    localStorage.removeItem('token')
}

// 请求拦截器，自动添加 Authorization 头
API.interceptors.request.use(config => {
    const token = getToken()
    if (token) config.headers.Authorization = `Bearer ${token}`
    return config
})

// 响应拦截器，处理 401 错误
API.interceptors.response.use(
    response => response, // 如果响应成功，直接返回响应
    error => {
        if (error.response && error.response.status === 401) {
            // 清除 token
            clearToken()

            // 跳转到登录页面
            router.push('/login')
        }
        return Promise.reject(error) // 继续抛出错误
    }
)

// 登录方法，获取 token 并存储
export function login(data) {
    return API.post('/auth/login', data, {
        headers: {
            'Content-Type': 'application/json',
        },
    }).then(response => {
        if (response.data && response.data.access_token) {
            setToken(response.data.access_token) // 存储 token
        }
        return response
    })
}

// 注册方法
export function register(data) {
    return API.post('/auth/register', data, {
        headers: {
            'Content-Type': 'application/json',
        },
    })
}

// 获取用户信息
export function getProfile() {
    return API.get('/users/me')
}

// 退出登录，清除 token
export function logout() {
    clearToken()
    return Promise.resolve();
}

// 获取标记点
export function getMarkers() {
    return API.get('/markers').then(response => response.data);
}

// 添加标记点
export function addMarker(data) {
    return API.post('/markers', data).then(response => response.data);
}

// 删除标记点
export function deleteMarker(markerId) {
    return API.delete(`/markers/${markerId}`).then(response => response.data);
}

// 获取户型数据
export function getFloors(marker_id) {
    return API.get(`/floors/markers/${marker_id}/floors`).then((response) => response.data);
}

// 新增 GeoJSON
export function addGeojson(data) {
    return API.post('/geojson/', data).then((response) => response.data);
}

// 新增 Floor
export function addFloor(data) {
    return API.post('/floors/', data).then((response) => response.data);
}

// 删除 Floor
export function deleteFloor(floor_id) {
    return API.delete(`/floors/${floor_id}`).then((response) => response.data);
}

// 删除 GeoJSON
export function deleteGeojson(geojson_id) {
    return API.delete(`/geojson/${geojson_id}`).then((response) => response.data);
}

// 获取 GeoJSON 数据
export function getGeojson(geojson_id) {
    return API.get(`/geojson/${geojson_id}`).then((response) => response.data);
}

// 获取结果列表
export function getResultsByFloor(floor_id) {
    return API.get(`/results/floors/${floor_id}/results`).then((response) => response.data);
}


export async function createResult(data, csvFile = null) {
    const formData = new FormData();

    let positionArray = [];
    if (Array.isArray(data.source.position)) {
        positionArray = data.source.position.map(Number);
    } else if (typeof data.source.position === 'string') {
        positionArray = data.source.position
            .split(/[,，]/)
            .map(item => parseFloat(item.trim()));
    }

    let pinnParams = undefined;
    if (data.simulation_type === 'pinn' && data.pinn_params) {
        const csvCoordsStr = data.pinn_params.csv_coordinates;
        console.log(csvCoordsStr);

        pinnParams = {};

        if (csvCoordsStr && typeof csvCoordsStr === 'string' && csvCoordsStr.trim() !== '') {
            pinnParams.csv_coordinates = csvCoordsStr
                .split(/[,，]/)
                .map(item => parseFloat(item.trim()))
                .filter(item => !isNaN(item)); // 防止 NaN 混入
        } else {
            pinnParams.csv_coordinates = null;
        }
    }


    formData.append('request', JSON.stringify({
        floor_id: Number(data.floor_id),
        simulation_type: data.simulation_type,
        description: data.description,
        source: {
            position: positionArray,
            strength: parseFloat(data.source.strength),
        },
        pinn_params: pinnParams,
        fvm_params: data.simulation_type === 'fvm' ? data.fvm_params : undefined,
    }));

    // 如果有 CSV 文件，则添加到 'csv_file' 字段
    if (csvFile) {
        formData.append('csv_file', csvFile);
    }

    try {
        const response = await API.post('/results/', formData);  // 等待请求完成
        return response.data;  // 返回响应数据
    } catch (error) {
        console.error('创建模拟结果失败:', error);
        throw error;  // 如果请求失败，抛出错误
    }
}

// 删除结果
export function deleteResultById(result_id) {
    return API.delete(`/results/${result_id}`).then((response) => response.data);
}

// gif
export function getGifById(gif_id) {
    return API.get(`/gifs/${gif_id}`).then((response) => {
        const data = response.data;
        if (data.gif && data.gif.gif_url && !data.gif.gif_url.startsWith('http')) {
            data.gif.gif_url = API.defaults.baseURL + data.gif.gif_url;
            console.log(data.gif.gif_url)
        }
        return data;
    });
}
