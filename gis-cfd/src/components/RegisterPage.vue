<template>
  <v-container class="register-container" fluid style="padding: 0;">
    <v-row justify="center" align="center" class="fill-height">
      <v-col cols="12" md="6" lg="4">
        <v-card class="register-card">
          <v-card-title class="register-title">注册</v-card-title>
          <v-card-text>
            <v-form v-model="valid" ref="form">
              <v-text-field
                v-model="username"
                label="用户名"
                :rules="usernameRules"
                required
              ></v-text-field>
              <v-text-field
                v-model="email"
                label="电子邮件"
                :rules="emailRules"
                required
              ></v-text-field>
              <v-text-field
                v-model="password"
                label="密码"
                type="password"
                :rules="passwordRules"
                required
              ></v-text-field>
              <v-select
                v-model="role"
                :items="roles"
                label="角色"
                required
              ></v-select>
              <v-btn
                color="primary"
                @click="register"
                :disabled="!valid"
                block
                class="mt-4"
              >
                注册
              </v-btn>
            </v-form>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { register } from '../utils/auth';

export default {
  data() {
    return {
      username: '',
      email: '',
      password: '',
      role: 'user',
      roles: ['user', 'admin'],
      valid: false,
      usernameRules: [(v) => !!v || '用户名不能为空'],
      emailRules: [(v) => /.+@.+\..+/.test(v) || '请输入有效的电子邮件地址'],
      passwordRules: [(v) => !!v || '密码不能为空'],
    };
  },
  methods: {
    register() {
      const registerData = {
        user_name: this.username,
        email: this.email,
        password: this.password,
        role: this.role,
      };

      register(registerData)
        .then((response) => {
          console.log('注册成功:', response.data);
          this.$router.push('/login');
        })
        .catch((error) => {
          console.error('注册失败:', error);
        });
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

.register-container {
  background-image: url('@/assets/map.jpg');
  background-size: 100% 100%;
  background-position: center;
  background-repeat: no-repeat;
  width: 100vw;
  height: 100vh;
}

.register-card {
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
  background-color: rgba(255, 255, 255, 0.9);
}

.register-title {
  justify-content: center;
  color: #1976D2;
  font-weight: bold;
  font-size: 24px;
  display: flex;
}
</style>

