<template>
  <v-app>
    <v-app-bar app flat color="#6b7785">
      <v-spacer></v-spacer>

      <template v-if="isAuthenticated">
        <v-btn text @click="$router.push('/')">首页</v-btn>
        <v-btn text @click="$router.push('/profile')">个人信息</v-btn>
        <v-btn text @click="logout">登出</v-btn>
      </template>
      <template v-else>
        <v-btn text @click="$router.push('/login')">登录</v-btn>
        <v-btn text @click="$router.push('/register')">注册</v-btn>
      </template>
    </v-app-bar>

    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script>
import { logout } from './utils/auth';

export default {
  name: 'App',
  data() {
    return {
      isAuthenticated: false, // 默认值为 false
    };
  },
  created() {
    // 应用加载时检查 token
    this.isAuthenticated = !!localStorage.getItem('token');
  },
  methods: {
    logout() {
      logout()
        .then(() => {
          this.isAuthenticated = false; // 手动更新状态
          this.$router.push('/login'); // 跳转到登录页面
        })
        .catch((error) => {
          console.error('登出失败:', error);
        });
    },
  },
  watch: {
    '$route'() {
      this.isAuthenticated = !!localStorage.getItem('token'); // 路由变化时更新状态
    },
  },
};
</script>

<style>
html, body, #app {
  margin: 0;
  padding: 0;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}
</style>
