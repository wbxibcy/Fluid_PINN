import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../components/HomePage.vue'
import LoginPage from '../components/LoginPage.vue'
import RegisterPage from '../components/RegisterPage.vue'
import ProfilePage from '../components/ProfilePage.vue'
import FloorsPage from '../components/FloorsPage.vue';
import ResultPage from '../components/ResultPage.vue';

const routes = [
    { path: '/', component: HomePage },
    { path: '/login', component: LoginPage },
    { path: '/register', component: RegisterPage },
    { path: '/profile', component: ProfilePage },
    { path: '/floors/:marker_id', name: 'FloorsPage', component: FloorsPage },
    { path: '/result/:floor_id', name: 'ResultPage', component: ResultPage, props: true,},
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

export default router
