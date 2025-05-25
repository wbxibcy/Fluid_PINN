<template>
  <v-container class="profile-container" fluid>
    <v-row justify="center" align="center" class="fill-height">
      <v-col cols="12" md="6" lg="4">
        <v-card>
          <v-card-title class="profile-title">个人信息</v-card-title>
          <v-card-text>
            <v-list dense>
              <v-list-item class="profile-item">
                <v-list-item-content class="profile-key">用户姓名</v-list-item-content>
                <v-list-item-content class="profile-value">{{ user.user_name }}</v-list-item-content>
              </v-list-item>
              <v-list-item class="profile-item">
                <v-list-item-content class="profile-key">电子邮件</v-list-item-content>
                <v-list-item-content class="profile-value">{{ user.email }}</v-list-item-content>
              </v-list-item>
              <v-list-item class="profile-item">
                <v-list-item-content class="profile-key">用户角色</v-list-item-content>
                <v-list-item-content class="profile-value">{{ user.role }}</v-list-item-content>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { getProfile } from '../utils/auth';

export default {
  data() {
    return {
      user: {},
    };
  },
  created() {
    getProfile()
      .then(response => {
        this.user = response.data.user;
      })
      .catch(error => {
        console.error('获取用户信息失败:', error);
        this.$router.push('/login');
      });
  },
};
</script>

<style scoped>
.profile-container {
  background-image: url('@/assets/xxx.jpg');
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  width: 100vw;
  height: 100vh;
  padding: 0 !important;
}

.fill-height {
  height: 100vh !important;
}

.profile-title {
  font-size: 1.5rem;
  color: #0d47a1; /* primary dark */
  justify-content: center !important;
  display: flex;
  width: 100%;
}

.v-list {
  padding: 0;
}

.profile-item {
  display: flex;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #e0e0e0;
  font-size: 1.125rem;
  line-height: 1.6;
}

.profile-key {
  width: 120px;         /* 固定宽度 */
  text-align: right;    
  font-weight: 600;
  color: #555;
  padding-right: 20px;  /* 键和值之间间隙 */
  flex-shrink: 0;
}

.profile-value {
  color: #333;
  flex: 1;              /* 占满剩余空间 */
  text-align: left;
  word-break: break-word;
}
</style>
