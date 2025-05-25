<template>
  <v-container class="login-container" fluid style="padding: 0;">
    <v-row justify="center" align="center" class="fill-height">
      <v-col cols="12" md="6" lg="4">
        <v-card class="login-card">
          <v-card-title class="login-title">登录</v-card-title>
          <v-card-text>
            <v-form v-model="valid" ref="form">
              <v-text-field
                v-model="email"
                label="邮箱"
                :rules="usernameRules"
                required
              ></v-text-field>
              <v-text-field
                v-model="password"
                label="密码"
                type="password"
                :rules="passwordRules"
                required
              ></v-text-field>
              <v-btn
                color="primary"
                @click="login"
                :disabled="!valid"
                block
                class="mt-4"
              >
                登录
              </v-btn>
            </v-form>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { login } from '../utils/auth'

export default {
  data() {
    return {
      email: '',
      password: '',
      valid: false,
      usernameRules: [(v) => !!v || '邮箱不能为空'],
      passwordRules: [(v) => !!v || '密码不能为空'],
    }
  },
  methods: {
    login() {
      const loginData = {
        email: this.email,
        password: this.password,
      }

      login(loginData)
        .then(() => {
          this.$router.push('/profile')
        })
        .catch(error => {
          console.error('登录失败:', error)
        })
    }
  }
}
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

.login-container {
  background-image: url('@/assets/earth.jpg');
  background-size: 100% 100%;
  background-position: center;
  background-repeat: no-repeat;
  width: 100vw;
  height: 100vh;
}

.login-card {
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
  background-color: rgba(255, 255, 255, 0.9);
}

.login-title {
  justify-content: center;
  color: #1976D2; /* 与 primary 按钮颜色一致 */
  font-weight: bold;
  font-size: 24px;
  display: flex;
}
</style>

